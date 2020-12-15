[TOC]

## ML算法

[机器学习哪些算法需要归一化](<https://blog.csdn.net/qq_34872215/article/details/88363504>)

### K近邻

K近邻方法是一种基本的分类和回归方法。利用训练数据集对特征空间做划分，并作为其分类的模型。

- 基本做法：

  先确定输入实例点的k个最近邻点，然后k个实例点预测输入实例点

- 三要素：k值选取、距离度量、决策规则

  k值小，学习的近似误差小，但是估计误差会增大，预测结果对近邻的实例点很敏感，容易过拟合。k值选取	可以采用交叉验证方法。

  分类问题：**多数表决决策规则**（选择这k个样本中出现最多的类别标记作为预测结果）、$L_p$度量

  回归问题：**平均法**（直接取平均或者基于距离远近加权平均，距离越近权值越大）

- k近邻搜索方法：**kd Tree**

  kd Tree 是一种二叉树，表示对k维空间的一种划分。

### 决策树

​	学习的三个步骤：特征选择、决策树的生成、决策树的剪枝

#### 特征选择问题（决定用哪个特征来划分特征空间）

- 准则：信息增益、信息增益比

- 信息增益
  $$
  \begin{aligned}
  H(D)	&=  - \sum_{k = 1}^K \frac{|C_k|}{|D|} \log_2 \frac{|C_k|}{D} \qquad 经验熵 \\
  H(D|A)	&= \sum_{i = 1}^n  \frac{|D_i|}{D} H(D_i)
  =- \sum_{i = 1}^n \frac{|D_i|}{|D|} 
  \{  
  - \sum_{k = 1}^K \frac{|D_{ik}|}{|D_i|} \log_2 \frac{| D_{ik} |}{|D_i|}
  \} \qquad 经验条件熵\\
  g(D,A) &= H(D) - H(D|A) \qquad  信息增益
  \end{aligned}
  $$
  其中，样本集合为`D`，类别个数为`K`，$C_k$是样本集合中属于第`k`类的样本子集，$D_i$是$D$中特征$A$取第$i$个值的样本子集，$D_{ik}$表示$D_i$中属于第$k$类的样本子集。

- 信息增益比
  $$
  g_R(D, A) = \frac{g(D,A)}{H_A(D)}
  $$
  其中
  $$
  H_A(D) = -\sum_{i = 1}^n \frac{|D_i|}{D} \log_2 \frac{|D_i|}{D} ,
  $$
  称为数据集$D$关于$A$的取值熵。

- Gini指数
  $$
  Gini(D) = 1 - \sum_{k = 1}^n ( \frac{|C_k|}{|D|} )^2
  $$
  特征A的Gini指数
  $$
  Gini(D, A) = \sum_{i=1}^n \frac{|D_i|}{D} Gini(D_i)
  $$

- 对比

<<<<<<< HEAD
<<<<<<< HEAD
![图片](Pictures\ML\decision_feature.png?raw=true)
=======
![图片](https://github.com/ren2504413601/2020interview/tree/main/Pictures/ML/decision_feature.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3
=======
![图片](https://github.com/ren2504413601/2020interview/tree/main/Pictures/ML/decision_feature.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

| 算法 | 支持模型   | 树结构 | 特征选择         | 连续值处理 | 缺失值处理 | 剪枝   |
| ---- | ---------- | ------ | ---------------- | ---------- | ---------- | ------ |
| ID3  | 分类       | 多叉树 | 信息增益         | 不支持     | 不支持     | 不支持 |
| C4.5 | 分类       | 多叉树 | 信息增益比       | 支持       | 支持       | 支持   |
| CART | 分类，回归 | 二叉树 | 基尼系数，均方差 | 支持       | 支持       | 支持   |



#### 决策树的生成

- `ID3`算法（最大信息增益准则选择特征）

  缺点：当一个属性可取的值较多时，可能在这个属性对应值下的样本只有一个或者很少，此时它的信息增益将会很高。

- `C4.5`算法（应用信息增益比选择特征）

- CART算法（应用最大Gini指数）

#### 决策树剪枝

​	避免过拟合。通过极小化决策树整体的损失函数来实现。一般有两种方法：预剪枝、后剪枝。

#### `CART`算法

​	决策树的生成就是递归的构建二叉决策树的过程。

- ##### 回归树

  最小化平方误差函数

- ##### 分类树

  最小化基尼指数`（Gini index）`

### 逻辑回归（Logistic regression）

应用极大似然估计法估计模型参数，似然函数的最大化可以考虑用梯度下降法或者倪牛顿法

### XGBoost

XGBoost是基于GBDT的一种算法或者说工程实现

GBDT 是一种基于boosting集成思想的加法模型。训练时采用前向分步算法进行贪婪式学习，每次迭代都学习一颗CART树（决策树）来拟合之前`t-1`棵树的预测结果与训练样本真实值的残差。

**提升树**

![图片](Pictures\boosting_tree.PNG?raw=true)

<<<<<<< HEAD
  ![图片](Pictures\square_loss_fit.PNG?raw=true)
=======
  ![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\square_loss_fit.PNG?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

**梯度提升**

损失函数选择平方误差函数和指数损失函数$L(y,f(x))=exp[-yf(x)]$(AdaBoost)时残差易于计算。一般损失函数可以利用损失函数的负梯度方向作为残差的近似值，拟合一个回归树。

<<<<<<< HEAD
![图片](Pictures\gradient_regression.png)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\gradient_regression.png)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

**Note:**XGBoost 与GBDT基本思想相同，但是作了一些优化。如默认的缺失值处理、损失函数加入二阶导数信息、正则项、列抽样，可以并行计算等。

- [GBDT/XGBoost 常见问题](https://mp.weixin.qq.com/s/AvK76Kx26mWFa1HJpoua_w)

- [深入理解XGBoost，优缺点分析，原理推导及工程实现](https://mp.weixin.qq.com/s/5zSLod4oyL4m6LADI6KC0Q)

- [算法理论+实战之LightGBM算法](https://mp.weixin.qq.com/s/pph0TsMoIM1G1mDmStdEeQ)

### `LightGBM`

`XGBoost`寻找最优分裂点的复杂度 = 特征数量 x 分裂点的数量 x 样本的数量

在`XGBoost` 基础上，` Lightgbm`里面的直方图算法就是为了减少分裂点的数量， `Lightgbm`里面的单边梯度抽样算法就是为了减少样本的数量， 而`Lightgbm`里面的互斥特征捆绑算法就是为了减少特征的数量。

问题：1 lightGBM相比GBDT和xgboost有什么区别？2 lightGBM重要性怎么评估？3 lightGBM节点怎么分裂的？

- [LightGBM’s documentation](<https://lightgbm.readthedocs.io/en/latest/index.html>)

### GBDT、XGBoost、LightGBM异同

#### [特征重要性](https://blog.csdn.net/yangxudong/article/details/53899260)

特征`j`的全局重要度通过特征`j`在单颗树中的重要度的平均值来衡量

XGBoost在训练过程中，通过Gini指数选择分离点的特征，一个特征被选中的次数越多，那么该特征评分也就越高。

#### GBDT和Xgboost的异同点

- GBDT损失函数只用了函数的一阶导数信息，XGBoost使用了二阶导数信息，效率更高、更准确。XGBoost损失函数欢加入模型复杂度的惩罚项（叶子节点个数+叶子节点权重的L2正则化），可以防止过拟合。
- GBDT采用CART作为基分类器，XGBoost可以自定义弱学习器类型（线性分类器），自定义损失函数，但必须二阶可导。
- GBDT在每轮迭代时使用全部的数据，Xgboost每一轮的训练中，不仅支持样本子采样（类似bagging），还支持列抽样,训练时只使用一部分的特征（随机森林策略）。
- 特征维度上的并行化。Xgboost每一轮的训练，各层节点可以并行训练，gbdt不行

#### LGB 和 XGBoost 异同

- 决策树算法。XGBoost使用的是pre-sorted算法，能够更精确的找到数据分隔点；LightGBM使用的是histogram算法（直方图算法），占用的内存更低，数据分隔的复杂度更低。
- 决策树生长策略。XGBoost采用的是level（depth）-wise生长策略，能够同时分裂同一层的叶子，从而进行多线程优化，不容易过拟合。LightGBM采用leaf-wise生长策略，每次从当前所有叶子中找到分裂增益最大（一般也是数据量最大）的一个叶子，然后分裂，如此循环。但会生长出比较深的决策树，产生过拟合。

### 集成学习

#### 集成学习的分类

boosting、bagging、blending、stacking等

- boosting

  - adaboost

    对分类正确的样本降低权重，对分类错误的样本升高或者保持权重不变

  - GBDT

    分步加法模型。每一颗树学习的是之前所有树的结论和残差

  - 基分类器的选择：决策树、神经网络（可以引入随机性）

- bagging

  - 随机森林

  - bagging集成后分类器的方差比基分类器的方差小。$\textcolor{red}{主要好处}$

- blending

- stacking

#### 方差偏差理论

一般情况下，我们评价模型性能时都会使用泛化误差。泛化误差越低，模型性能越好。泛化误差可分解为方差、偏差和噪声三部分。这三部分中，噪声是个不可控因素，它的存在是算法一直无法解决的问题，很难约减，所以我们更多考虑的是方差和偏差。

方差和偏差在泛化误差上可做如下分解，假设我们的预测值为`g(x)`，真实值为`f(x)`，则均方误差为
$$
E((g(x)−f(x))^2)
$$
这里假设不考虑噪声，g来代表预测值，f代表真实值，$\bar{g} = E(g(x))$代表算法的期望预测，则有如下表达：
$$
\begin{aligned}
E(g - f)^2	&= E(g^2 - 2gf + f^2) \\ 
			&= E(g^2) - \bar{g}^2 + (\bar{g} - f)^2 \\
			&= E(g^2) - 2 \bar{g}^2 + \bar{g}^2 + (\bar{g} - f)^2 \\
			&= \underbrace{E(g - \bar{g})^2}_{var(x)} + \underbrace{(\bar{g} - f)^2}_{bias^2(x)} 
\end{aligned}
$$
偏差与方差分别是用于衡量一个模型泛化误差的两个方面；

- 模型的偏差，指的是模型预测的期望值与真实值之间的差； 模型的方差，指的是模型预测的期望值与预测值之间的差平方和； 偏差用于描述模型的拟合能力； 方差用于描述模型的稳定性。

- Boosting 能提升弱分类器性能的原因是降低了偏差；Bagging 则是降低了方差；
- 我们认为方差越大（不稳定）的模型越容易过拟合。
- bagging和stacking中的基模型为强模型（偏差低方差高），boosting中的基模型为弱模型。

### Mulitlayer perceptron

**感知机**：

<<<<<<< HEAD
![图片](Pictures/perception.png?raw=true)

**单个隐藏层的感知机**：

![图片](Pictures\bp_nn.png?raw=true)

![图片](Pictures\bp_algorithm.png?raw=true)

**激励函数**：

![图片](Pictures\activattion_function.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\perception.png?raw=true)

**单个隐藏层的感知机**：

![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\bp_nn.png?raw=true)

![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\bp_algorithm.png?raw=true)

**激励函数**：

![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\activattion_function.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

**优化方法**：

- 梯度下降（Gradient descent）。缺点：在全部训练数据上训练，计算时间长，而且不一定全局最优

- 随机梯度下降（stochastic Gradient descent）。随机的优化某一条训练数据上的损失函数。这样更新速度会加快，但是缺点是甚至可能达不到局部最优。

- batch Gradient descent 。每次在一个batch上训练。

**学习率设置**：

- 指数衰减法

### 支持向量机(SVM)

### 逻辑回归（LR）

### LR、SVM异同

$$
\begin{aligned}
  J(\theta) &= - \frac{1}{n} \sum_{i = 1}^n [y_i \log h_{\theta}(x_i) + (1-y_i) \log (1 - h_{\theta}(x))]  \qquad 对数似然函数\\
  L(\omega, b, \alpha_i) &= \frac{1}{2} || \omega ||_2^2 - \sum_{i = 1}^n \{
  \alpha_i y_i (\omega \cdot x_i + b) -1 
  \}  \qquad 合页损失函数
  \end{aligned}
$$

- 相同点：
  - LR和SVM都是**分类**算法。
  - LR和SVM都是**监督学习**算法。
  - LR和SVM都是**判别模型**。
  - 如果不考虑核函数，LR和SVM都是**线性分类**算法，也就是说他们的分类决策面都是线性的。
    说明：LR也是可以用核函数的.但LR通常不采用核函数的方法。（**计算量太大**）

- 不同点：
  - LR 基于概率理论最大化似然函数估计参数，SVM基于几何间隔最大原理，最大化几何间隔
  - LR 对异常值敏感，SVM对异常值不敏感。支持向量机改变非支持向量样本并不会引起决策面的变化。逻辑回归中改变任何样本都会引起决策面的变化。
  - SVM会用核函数而LR一般不用核函数
  - 时间复杂度不同。当样本较少，特征维数较低时，SVM和LR的运行时间均比较短，SVM较短一些。准确率的话，LR明显比SVM要高。当样本稍微增加些时，SVM运行时间开始增长，但是准确率赶超了LR。
  - 对非线性问题的处理。LR主要靠特征构造，交叉特征、离散化特征。SVM也可以，主要通过Kernel函数。

  - LR使用log损失，SVM使用合页损失函数

### PCA

**主成分分析**（英语：**Principal components analysis**，**PCA**）是利用[正交变换](https://zh.wikipedia.org/wiki/正交变换)来对一系列可能相关的变量的观测值进行线性变换，从而投影为一系列线性不相关变量的值，这些不相关变量称为主成分（Principal Components）。具体地，主成分可以看做一个线性方程，其包含一系列线性系数来指示投影方向。`PCA`对原始数据的正则化或预处理敏感（相对缩放）。

主成分分析认为，沿某特征分布的数据的方差越大，则该特征所包含的信息越多，也就是所谓的主成分。 

- 最大方差理论
- 最小平方误差理论

`PCA` 只能处理线性数据的降维 (其本质上是线性变换)，仅是筛选方差最大的特征，去除特征之间的线性相关性，对于线性不可分的数据常常效果很差，因此提出了 `KPCA` 方法。
`KPCA`算法的思想如下，数据在低维度空间不是线性可分的，但是在高维度空间就可以变成线性可分的了。利用这个特点，将原始数据通过核函数（kernel）映射到高维度空间，再利用 `PCA`算法进行降维。 

<<<<<<< HEAD
![图片](Pictures\pca.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\pca.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3



### LDA

`LDA`是一种监督学习的降维技术，也就是说它的数据集的每个样本是有类别输出的。这点和`PCA`不同。`PCA`是不考虑样本类别输出的无监督降维技术。

$\color{red}LDA的思想可以用一句话概括，就是“投影后类内方差最小，类间方差最大”。$

- 最大化目标
  $$
  \max_{\omega} J(\omega) = \frac{ || \omega^T (\mu_1 - \mu_2) ||_2^2 }{D_1 + D_2},
  $$
  其中，$\omega$是单位向量，$D_1,D_2$分别代表两类投影后的方差

  

- [数学推导+纯Python实现机器学习算法27：LDA线性判别分析](https://mp.weixin.qq.com/s/KE9s0XFTx4Kj1kvAxhDVKA)

EM算法

[EM-最大期望算法](http://www.csuldw.com/2015/12/02/2015-12-02-EM-algorithms/)



### 聚类算法

根据聚类思想的划分：

- 基于划分的聚类：K-Means、K-medoids、CLARANS
- 基于层次的聚类：自底向上（AGNES）、自上向下的分裂方法（DIANA）
- 基于密度的聚类：DBSACN、OPTICS、BIRCH（CF-Tree）、CURE
- 基于网络的聚类：STING、WaveCluster
- 基于模型的聚类：EM、SOM、COBWB

#### Kmeans

<<<<<<< HEAD
![图片](Pictures\kmeans.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\kmeans.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

- `Kmeans`优缺点
  - 优点：计算复杂度`O(NKt)`接近线性，`N`是样本总数，`K`是聚类个数，`t`是迭代的轮数
  - 缺点：最终解可能不是全局最优而是局部最优。
- [数学推导+纯Python实现机器学习算法23：kmeans聚类](https://mp.weixin.qq.com/s/thjVrODB6p0e6O59lAc_Gw)

### 时间序列模型

#### ARMA和ARIMA

- AR(p)
  $$
  x_t = \phi_0 + \phi_1 x_{t-1} + \phi_2 x_{t-2} + \cdots + \phi_p x_{t-p} + \epsilon_t
  $$
  以前p期的序列值$x_{t-1},x_{t-2},...,x_{t-p}$为自变量、随机变量$X_t$的取值为因变量的线性回归模型

- MA(q)
  $$
  x_t = \mu + \epsilon_t - \theta_1 \epsilon_{t-1} - \theta_2 \epsilon_{t-2} + \cdots - \theta_q \epsilon_{t-p} + \epsilon_t
  $$
  随机变量$X_t$的取值与前各期取值无关，建立与前`q`期随机变量的扰动$\epsilon_{t-1},\epsilon_{t-2},...,\epsilon_{t-q}$的线性回归模型

- ARMA(p,q)

  自回归移动平均模型
  $$
  x_t = 
  \phi_0 + \phi_1 x_{t-1} + \phi_2 x_{t-2} + \cdots + \phi_p x_{t-p} + \epsilon_t + \epsilon_t - \theta_1 \epsilon_{t-1} - \theta_2 \epsilon_{t-2} + \cdots - \theta_q \epsilon_{t-p} + \epsilon_t
  $$

- ARIMA:差分运算+ ARMA
  
  `Idea:`许多非平稳序列差分后可以显示出平稳序列的性质

#### LSTM

`LSTM` 门结构

- `LSTM`是一种拥有三个“门”的特殊网络结构，包括遗忘门、输入门、输出门。

- 所谓“门”结构就是一个使用`sigmoid`神经网络和一个按位做乘法的操作，这两个操作合在一起就是一个门结构。

  [1]**遗忘门**的作用是让循环神经网络**忘记**之前没有用的信息。

  [2]循环神经网络忘记了部分之前的状态后，还需要从当前的输入**补充**最新的记忆。这个过程就是**输入门**完成的。

  [3]`LSTM`结构在计算得到新状态后需要产生当前时刻的输出，这个过程是由**输出门**完成的。

<<<<<<< HEAD
  ![图片](Pictures\NLP\LSTM_formula.jpg?raw=true)
=======
  ![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\NLP\LSTM_formula.jpg?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

公式定义
$$
\begin{aligned}
z	&= tanh(W_z[h_{t-1}, x_t])		&	(输入值)\\
i	&= simoid(W_i[h_{t-1}, x_t])	&	(输入门) \\
f	&= sigmoid(W_f[h_{t-1}, x_t])	&	(遗忘门) \\
o	&= sigmoid(W_o[h_{t-1}, x_t])	& 	(输出门) \\
c_t	&= f \cdot c_{t-1} + i \cdot z	&	(新状态)\\
h_t	&= o \cdot tanh c_t				& 	(输出)
\end{aligned}
$$

- SEIR模型



### 优化求解器(Optimizer)

[优化算法Optimizer比较和总结](https://zhuanlan.zhihu.com/p/55150256)

#### 梯度下降法(Gradient Descent)

- 梯度下降（Batch Gradient descent）
  $$
  \theta = \theta - \eta \cdot \bigtriangledown_{\theta} J(\theta)
  $$

- Mini-batch Gradient descent
  $$
  \theta = \theta - \eta \cdot \bigtriangledown_{\theta} J(\theta ; x^{(i:i+n)}; y^{(i:i+n)})
  $$

- 随机梯度下降(SGD)

$$
\theta = \theta - \eta \cdot \bigtriangledown_{\theta} J(\theta ; x^{(i)}; y^{(i)})
$$

​		

​		$\bigtriangledown_{\theta} J(\theta ; x^{(i)}; y^{(i)})$ 表示根据样本$(x^{(i)}, y^{(i)})$随机选择的一个梯度方向这里虽然引入了随机性和噪声，但期望仍然等于正确的梯度下降。

#### 动量优化法

- Momentum
  $$
  \begin{aligned}
  	v_t		& = \gamma v_{t-1} + \eta \bigtriangledown_{\theta} J(\theta) \\
  	\theta	& = \theta - v_t
  \end{aligned}
  $$
  使用动量(`Momentum`)的随机梯度下降法(`SGD`)，主要思想是引入一个积攒历史梯度信息动量来加速`SGD`。$\gamma$表示动力的大小，一般取值为0.9。

- Nesterov accelerated gradient 
  $$
  \begin{aligned}
  	v_t		& = \gamma v_{t-1} + \eta \bigtriangledown_{\theta} J(\theta - \gamma v_{t-1}) \\
  	\theta	& = \theta - v_t
  \end{aligned}
  $$

  - Nesterov动量梯度的计算在模型参数施加当前速度之后，因此可以理解为往标准动量中添加了一个校正因子。
  - 理解策略：在Momentun中小球会盲目地跟从下坡的梯度，容易发生错误。所以需要一个更聪明的小球，能提前知道它要去哪里，还要知道走到坡底的时候速度慢下来而不是又冲上另一个坡。计算$\theta - \gamma v_{t-1}$可以表示小球下一个位置大概在哪里。从而可以提前知道下一个位置的梯度，然后使用到当前位置来更新参数

#### 自适应学习率优化算法

- Adagard
  $$
  \begin{aligned}
  g_t			 &= \bigtriangledown_{\theta} J(\theta_t) \\
  \theta_{t+1} &= \theta_{t} -\frac{\eta}{\sqrt{G_t + \epsilon}} \bigodot g_t \\
  G_t 		 &= \sum_{t' = 1}^t (g_{t'})
  \end{aligned}
  $$

  - AdaGrad算法，独立地适应所有模型参数的学习率，缩放每个参数反比于其所有梯度历史平均值总和的平方根。具有代价函数最大梯度的参数相应地有个快速下降的学习率，而具有小梯度的参数在学习率上有相对较小的下降。$\epsilon$是一个取值很小的数（一般为1e-8）为了避免分母为0。
  - 从表达式可以看出，对出现比较多的类别数据，Adagrad给予越来越小的学习率，而对于比较少的类别数据，会给予较大的学习率。因此Adagrad适用于数据稀疏或者分布不平衡的数据集。
  - Adagrad 的主要优势在于不需要人为的调节学习率，它可以自动调节；缺点在于随着迭代次数增多，学习率会越来越小，最终会趋近于0。

- RMSProp
  $$
  \begin{aligned}
  E[g^2]_t	&= \gamma E[g^2]_{t-1} + (1-\alpha) g_t^2 \\
  \theta_{t+1}&= \theta_{t} -\frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \bigodot g_t  \\
  
  \end{aligned}
  $$

  - $E[g^2]_t$表示前`t`次的梯度平方的均值。由于取了个加权平均，避免了学习率越来越低的的问题，而且能自适应地调节学习率。
  - RMSProp算法修改了AdaGrad的梯度积累为指数加权的移动平均，使得其在非凸设定下效果更好

- Adam
  $$
  \begin{aligned}
  m_t		&= \beta_1 m_{t-1} + (1-\beta_1)g_t \\
  v_t 	&= \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \\
  \hat{m_t}&= \frac{m_t}{1-\beta_1^t}, \quad \hat{v_t} = \frac{v_t}{\beta_2^t} \\
  \theta_{t+1}&= \theta_t - \frac{\eta}{\sqrt{\hat{v_t}} + \epsilon} \hat{m_t} 
  \end{aligned}
  $$

  - 其中，$m_t$和$v_t$分别为一阶动量项和二阶动量项。$\beta_1, \beta_2$为动力值大小通常分别取0.9和0.999;$\hat{m_t}, \hat{v_t}$分别为各自的修正值。

  - Adam中动量直接并入了梯度一阶矩（指数加权）的估计。其次，相比于缺少修正因子导致二阶矩估计可能在训练初期具有很高偏置的RMSProp，Adam包括偏置修正，修正从原点初始化的一阶矩（动量项）和（非中心的）二阶矩估计。

- AdaDelta
  $$
  \begin{aligned}
  E[g^2]_t	&= \gamma E[g^2]_{t-1} + (1-\alpha) g_t^2 \\
  \Delta \theta_t &= -\frac{\sqrt{\sum_{i=1}^{t-1} \Delta \theta_i}}{\sqrt{E[g^2]_t + \epsilon}} \\
  \theta_{t+1}&= \theta_{t} + \Delta \theta_t   \\
  \end{aligned}
  $$
  

  - AdaGrad算法和RMSProp算法都需要指定全局学习率，AdaDelta算法结合两种算法每次参数的更新步长即,AdaDelta不需要设置一个默认的全局学习率。

[各种优化器Optimizer的总结与比较](<https://blog.csdn.net/weixin_40170902/article/details/80092628>)

### 欠拟合、过拟合处理常用方法

#### 欠拟合

- 添加其他特征项。组合、泛化、相关性、上下文特征、平台特征等特征是特征添加的重要手段，有时候特征项不够会导致模型欠拟合。
- 添加多项式特征。例如将线性模型添加二次项或三次项使模型泛化能力更强。例如，FM（Factorization Machine）模型、FFM（Field-aware Factorization Machine）模型，其实就是线性模型，增加了二阶多项式，保证了模型一定的拟合程度。
- 可以增加模型的复杂程度。
- 减小正则化系数。正则化的目的是用来防止过拟合的，但是现在模型出现了欠拟合，则需要减少正则化参数。

#### 过拟合

- 基于模型的方法

  - 简化模型（比如将非线性模型改为线性的模型）

  - 集成学习：boosting、Bagging

  - Early stopoing。训练数据分成`train data`和`validation data`，训练误差降低但是验证误差增高，则停止

  - Dropout

    [深度学习中Dropout原理解析](https://blog.csdn.net/program_developer/article/details/80737724)

    - 指在深度网络的训练中， 以一定的概率随机地 “临时丢弃”一部分神经元节点 
    - 说的简单一点就是：我们在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征

  - `Batch Normalization `

    [深入理解Batch Normalization批标准化](https://www.cnblogs.com/guoyaohua/p/8724433.html)

    对于每个隐层神经元，把逐渐向非线性函数映射后向取值区间极限饱和区靠拢的输入分布强制拉回到均值为0方差为1的比较标准的正态分布，使得非线性变换函数的输入值落入对输入比较敏感的区域，**以此避免梯度消失问题**

  - 正则化。在误差目标函数中增加一个描述网络复杂度的部分
    $$
    E = \lambda \frac{1}{m} \sum_{k = 1}^m E_k + (1 - \lambda) \sum_{i} \omega_i^2
    $$

- 基于数据的方法
  
  - 数据扩充（Data Augmentation）

### L1和L2正则

L1正则解空间是稀疏的。

*L1*正则是拉普拉斯先验,*L2*是高斯先验

- 解空间形状

  L2正则的解空间是圆形，L1正则的解空间是多边形。多边形的解空间更容易在尖角处于等高线碰撞处稀疏解

<<<<<<< HEAD
  ![图片](Pictures\ML\regular_space.png?raw=true)
=======
  ![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\ML\regular_space.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3


### 超参数调优方法

- 网格搜索
- 随机搜索
- 贝叶斯优化

#### 归一化的选择

概率模型不需要归一化，因为它们不关心变量的值，而是关心变量的分布和变量之间的条件概率。像svm、线性回归之类的最优化问题就需要归一化。决策树属于前者。



## DL算法

[秋招算法岗面试知识，欢迎补充](<https://www.nowcoder.com/discuss/502430?type=0&order=7&pos=208&page=4&source_id=discuss_center_0&channel=1009>)

### [梯度消失、爆炸原因及其解决方法](https://zhuanlan.zhihu.com/p/33006526?from_voters_page=true)

- 梯度消失、梯度爆炸：
  - 两种情况下梯度消失经常出现，一是在深层网络中，二是采用了不合适的损失函数，比如sigmoid。
  - 梯度爆炸一般出现在深层网络和权值初始化值太大的情况下。

- 解决方案
  - 预训练加微调
  - 梯度剪切、权重正则（针对梯度爆炸）
  - 使用不同的激活函数
  - 使用batchnorm
  - 使用残差结构
  - 使用LSTM网络(门结构保留之前训练的信息，用于解决梯度消失)

### [分类问题为什么使用交叉熵(Cross Entry)作为损失函数](https://www.cnblogs.com/shine-lee/p/12032066.html)

[为什么分类问题的损失函数采用交叉熵而不是均方误差MSE？](https://zhuanlan.zhihu.com/p/104130889)

- 神经网络反向传播或者LR计算极大似然估计函数时（最优化问题），需要计算损失函数关于`w`的导数，均方误差损失函数关于$w$的导数包含sigmoid函数的导数$\sigma^{\prime} (z) $而交叉熵函数不包含。sigmoid函数的导数$\sigma^{\prime} (z) $在输出接近 0 和 1 的时候是非常小的，故导致在使用最小均方差Loss时，模型参数w会学习的非常慢。

- 假设给定输入为x，label为y，其中y的取值为0或者1，是一个分类问题。我们要训练一个最简单的Logistic Regression来学习一个函数f(x)使得它能较好的拟合label，如下图所示。

<<<<<<< HEAD
![图片](Pictures\ML\LR_loss.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\ML\LR_loss.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

其中$z(x) = \omega \cdot x + b, \quad \sigma(z) = \frac{1}{1+e^{-z}}$。均方差（Mean square error），和交叉熵(Cross entry error)损失如下：
$$
\begin{aligned}
L_{mse} 	&= \frac{1}{2} (\sigma(z) - y)^2 \\
L_{cee}		&= -(y \ln (z) + (1-y) \ln (1-z))
\end{aligned}
$$
对应导数，根据链式法则$ \frac{\partial L}{\partial \omega} = \frac{\partial L}{\partial \sigma(z)}  \cdot \frac{\partial  \sigma(z)}{\partial z} \cdot  \frac{\partial z}{\partial  \omega} $
$$
\begin{aligned}
\frac{\partial L_{mse}}{\partial \omega} &= (\sigma(z) - y) \cdot \sigma^{\prime}(z) \cdot x  \\
\frac{\partial L_{cee}}{\partial \omega} &= (- (\frac{y}{\sigma(z)} + \frac{1-y}{1-\sigma(z)})) \cdot \sigma^{\prime}(z) \cdot x \\
										&= (- (\frac{y}{\sigma(z)} + \frac{1-y}{1-\sigma(z)})) \cdot \sigma(z) \cdot (1- \sigma(z)) \cdot x \\
										&= (\sigma(z) - y) \cdot x
\end{aligned}
$$


## NLP 算法

#### 词袋模型和TF-IDF

- 词袋模型：one-hot编码

- tf-idf（词频-逆文档频率）

  Idea：如果某个词或短语在一篇文章中出现的频率高（TF大），并且在其他文章中很少出现(IDF小)，则认为此词或者短语具有很好的类别区分能力，适合用来分类。
  $$
  w_{tf-idf} = w_{tf} \cdot \log \frac{1}{w_{df}}
  $$
  $w_{tf}$是文档的词频，$w_{idf}$是包含该单词的所有文档的总频率。

- CountVectorizer

  不同于词袋模型的地方在于其不光对单词索引编码，还统计每一个单词出现的次数

- TruncatedSVD

  类似于PCA，不同的是TSVD直接处理样本矩阵X，而不是X的协方差矩阵$X^T X$(数据标准化后的协方差矩阵)。

#### 主题模型 : `LDA`(隐含狄利克雷分布Latent Dirichlet Allocation）

#### 词嵌入（`word embedding`）

- `word2vec` : `Continues Bag of Words(CBOW) 、Skip-gram`  

  CBOW的目标是根据上下文出现的词语来预测当前词的生成概率 。$\textcolor{red}{类似于完型填空}$

  Skip-gram是根据当前词来预测上下文中各词的生成概率 

<<<<<<< HEAD
![图片](Pictures\NLP\word2vector.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\NLP\word2vector.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

#### 循环神经网络RNN

[RNN](https://blog.csdn.net/zhaojc1995/article/details/80572098>)

[RNN - LSTM - GRU](https://zhuanlan.zhihu.com/p/60915302)

- 标准RNN

<<<<<<< HEAD
![图片](Pictures\NLP\rnnbp.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\NLP\rnnbp.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

对于$t$时刻：
$$
h^{(t)}=\phi(Ux^{(t)}+Wh^{(t-1)}+b)
$$
其中$\phi()$为激活函数，一般会选择tanh函数，$b$为偏置。

$t$时刻的输出为：
$$
o^{(t)}=Vh^{(t)}+c
$$
模型的预测输出为：
$$
\widehat{y}^{(t)}=\sigma(o^{(t)})
$$
其中$\sigma$为激活函数，通常RNN用于分类，故这里一般用softmax函数。

- 长短时间记忆网路LSTM

  - `LSTM` 门结构
    - `LSTM`是一种拥有三个“门”的特殊网络结构，包括遗忘门、输入门、输出门。
    - 所谓“门”结构就是一个使用`sigmoid`神经网络和一个按位做乘法的操作，这两个操作合在一起就是一个门结构。

  [1]**遗忘门**的作用是让循环神经网络**忘记**之前没有用的信息。

  [2]循环神经网络忘记了部分之前的状态后，还需要从当前的输入**补充**最新的记忆。这个过程就是**输入门**完成的。

  [3]`LSTM`结构在计算得到新状态后需要产生当前时刻的输出，这个过程是由**输出门**完成的。

  - 公式定义

<<<<<<< HEAD
  ![图片](Pictures/NLP/LSTM_formula.jpg?raw=true)
=======
  ![](https://github.com/ren2504413601/2020interview/tree/main/Pictures/NLP/LSTM_formula.jpg?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3
  $$
  \begin{aligned}
  z	&= tanh(W_z[h_{t-1}, x_t])		&	(输入值)\\
  i	&= simoid(W_i[h_{t-1}, x_t])	&	(输入门) \\
  f	&= sigmoid(W_f[h_{t-1}, x_t])	&	(遗忘门) \\
  o	&= sigmoid(W_o[h_{t-1}, x_t])	& 	(输出门) \\
  c_t	&= f \cdot c_{t-1} + i \cdot z	&	(新状态)\\
  h_t	&= o \cdot tanh c_t				& 	(输出)
  \end{aligned}
  $$
  其中， $h， c, t$分别表示输出，状态和输入

## 计算机网络

[计算机网络常见面试题](<https://www.nowcoder.com/discuss/415786>)

#### 浏览器输入网址之后发生了什么

1. DNS域名解析
2. HTTP协议生成请求报文
3. TCP协议将请求报文分割成报文段，进行可靠传输
4. IP协议进行分组转发
5. TCP协议重组请求报文
6. HTTP协议对请求进行处理

#### 网络体系（OSI七层、TCP/IP四层、五层）

<<<<<<< HEAD
#### ![图片](Pictures\computer_network\network_system.png)[TCP三次握手四次挥手](https://www.cnblogs.com/zhuzhenwei918/p/7465467.html)

- TCP三次握手

![图片](Pictures\computer_network\TCP_threeway_handshake.png?raw=true)
=======
#### ![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\computer_network\network_system.png)[TCP三次握手四次挥手](https://www.cnblogs.com/zhuzhenwei918/p/7465467.html)

- TCP三次握手

![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\computer_network\TCP_threeway_handshake.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

第一次握手：

客户端发送一个TCP的SYN标志位置1的包指明客户打算连接的服务器的端口，以及初始序号X,保存在包头的序列号(Sequence Number)字段里

第二次握手 ：

服务器发回确认包**(ACK)应答**。即**SYN标志位和ACK标志位均为1**同时，将确认序号(Acknowledgement Number)设置为客户的I S N加1以.即X+1。

第三次握手：

客户端再次发送确认包(ACK) SYN标志位为0,**ACK标志位为1**.并且把服务器发来ACK的序号字段+1,放在确定字段中发送给对方.并且在数据段放写ISN的+1

$\textcolor{red}{即SYN就是询问： 你能听得到吗？  ACK就是回到： 我能听得到啊。}$

**为什么是三次握手**？

- 三次握手才能阻止重复历史连接的初始化

- 三次连接才可以同步双方的初始序列号
- 三次握手可以避免资源浪费

**为什么不是两次或者四次握手**？

两次握手，无法阻止历史连接的建立，会造成双方资源的浪费，也无法可靠的同步双方序列号。

四次握手：三次握手已经理论上最少可靠连接建立，所以不需要使用更多的通信次数。

如果两次，那么B无法确定B的信息A是否能收到，所以如果B先说话，可能后面的A都收不到，会出现问题 。

如果四次，那么就造成了浪费，因为在三次结束之后，就已经可以保证A可以给B发信息，A可以收到B的信息； B可以给A发信息，B可以收到A的信息

<<<<<<< HEAD
![图片](Pictures\computer_network\TCP_threeway_handshake.png?raw=true)

  - TCP四次挥手

![图片](Pictures\computer_network\tcp四次挥手.png?raw=true)

  四次挥手：

![图片](Pictures\computer_network\tcp四次挥手解释.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\computer_network\TCP_threeway_handshake.png?raw=true)

  - TCP四次挥手

![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\computer_network\tcp四次挥手.png?raw=true)

  四次挥手：

![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\computer_network\tcp四次挥手解释.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

  A:“喂，**我不说了 (FIN)**。”A->FIN_WAIT1

  B:“我知道了**(ACK)**。等下，**上一句还没说完**。Balabala…..（**传输数据**）”B-  >CLOSE_WAIT | A->FIN_WAIT2

  B:”好了，说完了，我也不说了（**FIN**）。”B->LAST_ACK

  A:”我知道了（**ACK**）。”A->TIME_WAIT | B->CLOSED

  A**等待2MSL**,保证B收到了消息,否则重说一次”我知道了”,A->CLOSED

  这样，通过四次挥手，可以把该说的话都说完，并且A和B都知道自己没话说了，对方也没花说了，然后就挂掉电话（**断开链接**）了 。

#### TCP和UDP区别

HTTP/3 把 HTTP 下层的 TCP 协议改成了 UDP！

- 连接：UDP是无连接的，发送数据之前无需建立连接，发送数据结束后也无需释放连接。TCP是面向连接的，在发送数据之前需要通过三次握手建立连接，发送数据结束后需要通过四次挥手释放连接。

  TCP可靠连接的实现：滑动窗口、选择确认、超时重传

- 交付：UDP使用尽最大努力交付，即不保证可靠交付，主机不需要维持复杂的连接状态表。TCP提供可靠交付，即通过TCP连接传送的数据，无差错、不丢失、不重复，并且按序到达。

- 数据：UDP是面向报文的，UDP对于应用层交下来的报文，添加首部后就向下交给IP层，既不合并，也不拆分，一次交付一个完整的报文。TCP是面向字节流的，虽然应用程序和TCP的交互是一次一个数据块（大小不等），但TCP把应用程序交下来的数据看出成一连串无结构的字节流。TCP不保证接收方应用程序所收到的数据块和发送方应用程序所发出的数据块具有对应大小关系。 

- 通信双方：UDP支持一对一、一对多、多对一、多对多的交互通信。TCP连接是点对点（一对一），每一条TCP连接只能有两个端点。 

- 拥塞：UDP没有拥塞控制，网络的拥塞不会使源主机的发送速率降低。TCP通过慢开始、拥塞避免、快重传、快恢复等算法进行拥塞控制。 

- 首部：UDP首开销小，只有8个字节。TCP首部是20个字节。

#### TCP可靠性连接的实现

序列号、确认应答、重发控制、连接管理以及窗口控制等机制实现可靠性传输的

- 针对数据包丢失情形，使用**重传机制**解决

- 针对数据包往返时间长，通信效率就会变低，使用**滑动窗口**解决
- 发送方不能无脑的发送数据给接收方，要考虑接收方的处理能力。为了避免接收方处理不过来，TCP提供**流量控制**机制解决。
- **拥塞控制**，目的是为了避免发送方的数据填满整个网络。

<<<<<<< HEAD
![图片](Pictures\computer_network\TCP可靠性传输.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\computer_network\TCP可靠性传输.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

#### HTTP和HTTPS的区别

- 端口 ：HTTP的URL由“http://”起始且默认使用端口80，而HTTPS的UR由“https://”起始且默认使用端口443。
- 安全性和资源消耗： HTTP协议运行在TCP之上，所有传输的内容都是明文，客户端和服务器端都无法验证对方的身份。HTTPS是运行在SSL/TLS之上的HTTP协议，SSL/TLS 运行在TCP之上。所有传输的内容都经过加密，加密采用对称加密，但对称加密的密钥用服务器方的证书进行了非对称加密。所以说，HTTP 安全性没有 HTTPS高，但是 HTTPS 比HTTP耗费更多服务器资源。 
  - 对称加密：密钥只有一个，加密解密为同一个密码，且加解密速度快，典型的对称加密算法有DES、AES等。
  - 非对称加密：密钥成对出现（且根据公钥无法推知私钥，根据私钥也无法推知公钥），加密解密使用不同密钥（公钥加密需要私钥解密，私钥加密需要公钥解密），相对对称加密速度较慢，典型的非对称加密算法有RSA、DSA等。  

- HTTP 连接建立相对简单， TCP 三次握手之后便可进行 HTTP 的报文传输。而 HTTPS 在 TCP三次握手之后，还需进行 SSL/TLS 的握手过程，才可进入加密报文传输。

- HTTPS需要向CA（证书权威机构）申请数字证书，来保证服务器的身份是可信的。

<<<<<<< HEAD
![图片](Pictures\computer_network\https加密.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\computer_network\https加密.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

 证书验证，客户端发送一个证书请求个服务器端，服务器端返回证书，客户端对证书进行验证。 

 交换密钥，使用非对称加密，客户端使用公钥进行加密，服务器端使用密钥解密。 

 交换数据，使用对称加密的方式对数据进行加密，然后进行传输

#### HTTP协议中GET和POST方式的区别

参数位置、参数长度、参数编码、TCP数据包

- 参数位置：GET方法参数位置包含在URL，POST方法参数包含在请求主体

- 参数长度：GET方法的URL长度有限度，POST长度没有限制

- 参数编码：GET方法参数编码是ASCII码，POST没有限制

- TCP数据包：GET方法产生一个TCP数据包，把首部和数据一起发送，POST方法产生两个TCP数据包，先发首部，服务器响应后再发数据

## 操作系统

#### 进程、线程

[面试题之操作系统](https://www.nowcoder.com/discuss/468422?page=3)

进程是资源分配的最小单位，线程是程序执行（CPU调度）的最小单位（资源调度的最小单位）。一个线程只能属于一个进程，而一个进程可以有多个线程，至少有一个线程。

$$\textcolor{red}{并行和并发}$$

**并发就是在一段时间内，多个任务都会被处理；**但在某一时刻，只有一个任务在执行。单核处理器可以做到并发。比如有两个进程A和B，A运行一个时间片之后，切换到B，B运行一个时间片之后又切换到A。因为切换速度足够快，所以宏观上表现为在一段时间内能同时运行多个程序。  

**并行就是在同一时刻，有多个任务在执行。**这个需要多核处理器才能完成，在微观上就能同时执行多条指令，不同的程序被放到不同的处理器上运行，这个是物理上的多个进程同时进行。

- 进程和线程区别
  - 线程的优缺点
  
    - 优点：各个线程之间可以并发执行（线程能减少并发执行的时间和空间开销）；各个线程之间可以共享地址空间和文件等资源；
  
    - 缺点：当进程中的一个线程奔溃时，会导致其所属进程的所有线程奔溃。
  
    
  
- 进程的生命周期的三种状态

  ready(就绪),running(运行),wait(等待).

- 进程调度算法

  先来先服务、短作业优先、最短剩余时间优先、时间片轮转、优先级调度

- 线程的生命周期及五种基本状态（java）

  new、runnable、running、blocked、dead

  新建状态、就绪状态、运行状态、阻塞状态（等待阻塞、同步阻塞、其他阻塞）、死亡状态

  **进程状态的变迁:**

  ![img](https://mmbiz.qpic.cn/mmbiz_png/J0g14CUwaZcvw4t9kicec370n3cvX2JS9gjKOC2IyZwLJXMcqzgvpKia0u1ezepiawX0iaFkrvsLeV6qsHplv5grnw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

- 进程同步方式
  
  临界区、互斥量、信号量、管程
  
- 进程之间的通信方式
  
  - 管道(pipe)
  - 信号量(semophonre)
  - 信号(signal)
  - 消息队列(message queue)
  - 共享内存(shared memory)
  - 套接字（Socket）
  
- 线程同步
  - 锁机制：包括互斥锁/量（mutex）、读写锁（reader-writer lock）、自旋锁（spin lock）、条件变量（condition）
  - 信号量机制(Semaphore)
  - 信号机制(Signal)
  - 屏障（barrier）
  
- 进程之间私有和共享的资源
  - 私有：地址空间、堆、全局变量、栈、寄存器
  - 共享：代码段，公共数据，进程目录，进程 ID
  
- 线程之间私有和共享的资源
  - 私有：线程栈，寄存器，程序计数器
  - 共享：堆，地址空间，全局变量，静态变量

- fork时子进程获得父进程代码和数据段、共享库、堆和栈的复制，所以变量的地址（当然是虚拟地址）也是一样的

#### 死锁

死锁是一个进程集合中的多个进程因为竞争资源，造成的互相等待现象。

- 原因

  系统资源不足

  进程进行推行的顺序不合适

  资源分配不当等

- 死锁产生的必要条件

  互斥条件（Mutual exclusion ）：资源不能被共享，只能由一个进程使用。

   请求与保持条件（Hold and wait ）：已经得到资源的进程可以再次申请新的资源。

  非剥夺条件（No pre-emption ）：已经分配的资源不能从相应的进程中被强制地剥夺。

  循环等待条件（Circular wait ）：系统中若干进程组成环路，改环路中每个进程都在等待相邻进程正占用的资源。

  这四个条件是死锁的必要条件，只要系统发生死锁，这些条件必然成立，而只要上述条件之一不满足，就不会发生死锁。

- 处理死锁的基本方法

  鸵鸟策略（即直接忽略死锁）、死锁预防、死锁避免、死锁检测和恢复

#### [僵尸进程](https://blog.csdn.net/han_shi_lei/article/details/90046912)

**僵尸进程**是当子进程比父进程先结束，而父进程又没有回收子进程（调用wait/waitpid），释放子进程占用的资源，此时子进程将成为一个僵尸进程。会造成内存泄漏。

**孤儿进程**就是说一个父进程退出，而它的一个或多个子进程还在运行，那么这些子进程将成为孤儿进程。孤儿进程将被 `init` 进程(进程`ID`为`1`的进程)所收养，并由 `init` 进程对它们完成状态收集工作。因为孤儿进程会被 `init` 进程收养，所以孤儿进程不会对系统造成危害。  

**守护进程**是运行在后台的一种特殊进程，它是独立于控制终端的，并周期性地执行某些任务。

$$\textcolor{red}{僵尸进程如何避免？}$$

1、让僵尸进程的父进程来回收，父进程每隔一段时间来查询子进程是否结束并回收，调用wait()或者waitpid(),通知内核释放僵尸进程 。

2、采用信号SIGCHLD通知处理，并在信号处理程序中调用wait函数 。

3、让僵尸进程变成孤儿进程，就是让他的父亲先死。

4、如果父进程很忙，那么可以用signal函数为SIGCHLD安装handler，因为子进程结束后， 父进程会收到该信号，可以在handler中调用wait回收。

## 数据库

[数据库基础知识](https://www.nowcoder.com/discuss/434757?type=post&order=time&pos=&page=1&channel=1009&source_id=search_post>)

### 索引

- 索引的优点：

  第一，通过创建唯一性索引，可以保证数据库表中每一行数据的唯一性。

  第二，可以大大加快数据的检索速度，这也是创建索引的最主要的原因。

  第三，可以加速表和表之间的连接，特别是在实现数据的参考完整性方面特别有意义。

  第四，在使用分组和排序子句进行数据检索时，同样可以显著减少查询中分组和排序的时间。

  第五，通过使用索引，可以在查询的过程中，使用优化隐藏器，提高系统的性能。

- 索引的缺点：
  第一，创建索引和维护索引要耗费时间，这种时间随着数据量的增加而增加。
  第二，索引需要占物理空间，除了数据表占数据空间之外，每一个索引还要占一定的物理空间，如果要建立聚簇索引，那么需要的空间就会更大。
  第三，当对表中的数据进行增加、删除和修改的时候，索引也要动态的维护，这样就降低了数据的维   护速度。



## 算法和数据结构

### 排序算法

| 排序算法 | 平均时间复杂度 | 最差时间复杂度 | 空间复杂度 | 数据对象稳定性       |
| -------- | -------------- | -------------- | ---------- | -------------------- |
| 冒泡排序 | O(n2)          | O(n2)          | O(1)       | 稳定                 |
| 选择排序 | O(n2)          | O(n2)          | O(1)       | 数组不稳定、链表稳定 |
| 插入排序 | O(n2)          | O(n2)          | O(1)       | 稳定                 |
| 快速排序 | O(n*log2n)     | O(n2)          | O(log2n)   | 不稳定               |
| 堆排序   | O(n*log2n)     | O(n*log2n)     | O(1)       | 不稳定               |
| 归并排序 | O(n*log2n)     | O(n*log2n)     | O(n)       | 稳定                 |
| 希尔排序 | O(n*log2n)     | O(n2)          | O(1)       | 不稳定               |
| 计数排序 | O(n+m)         | O(n+m)         | O(n+m)     | 稳定                 |
| 桶排序   | O(n)           | O(n)           | O(m)       | 稳定                 |
| 基数排序 | O(k*n)         | O(n2)          |            | 稳定                 |

- 均按从小到大排列
- k：代表数值中的 “数位” 个数
- n：代表数据规模
- m：代表数据的最大值减最小值

笔试常考：

​	元素的**移动次数**与关键字的初始排列次序无关的是：基数排序

​	元素的**比较次数**与初始序列无关是：选择排序

​	算法的**时间复杂度**与初始序列无关的是：直接选择排序

### 查找算法

| 查找算法                     | 平均时间复杂度   | 空间复杂度 | 查找条件   |
| ---------------------------- | ---------------- | ---------- | ---------- |
| 顺序查找                     | O(n)             | O(1)       | 无序或有序 |
| 二分查找（折半查找）         | O(log2n)         | O(1)       | 有序       |
| 插值查找                     | O(log2(log2n))   | O(1)       | 有序       |
| 斐波那契查找                 | O(log2n)         | O(1)       | 有序       |
| 哈希查找                     | O(1)             | O(n)       | 无序或有序 |
| 二叉查找树（二叉搜索树查找） | O(log2n)         |            |            |
| 红黑树                       | O(log2n)         |            |            |
| 2-3树                        | O(log2n - log3n) |            |            |
| B树/B+树                     | O(log2n)         |            |            |

### 树结构

#### 二叉查找树

二叉查找树的特点就是**左子树的节点值比父亲节点小，而右子树的节点值比父亲节点大**

#### 平衡二叉树（AVL树）

平衡二叉树是为了解决二叉查找树退化成一颗链表，平衡树具有如下特点：

- 具有二叉查找树的全部特性。

- 每个节点的左子树和右子树的高度差至多等于1。

#### 红黑树

如果在那种插入、删除很频繁的场景中，平衡树需要频繁着进行调整，这会使平衡树的性能大打折扣，为了解决这个问题，于是有了红黑树，红黑树具有如下特点：

- 具有二叉查找树的特点。
- 根节点是黑色的；
- 每个叶子节点都是黑色的空节点（NIL），也就是说，叶子节点不存数据。
- 任何相邻的节点都不能同时为红色，也就是说，红色节点是被黑色节点隔开的。
- 每个节点，从该节点到达其可达的叶子节点是所有路径，都包含相同数目的黑色节点。

**Note**: [有了二叉查找树、平衡树为啥还需要红黑树？](<https://baijiahao.baidu.com/s?id=1636557496125304849&wfr=spider&for=pc>)

平衡树是为了解决二叉查找树退化为链表的情况，而红黑树是为了解决平衡树在插入、删除等操作需要频繁调整的情况。

通过对任何一条从根到叶子的路径上各个节点着色方式的限制，红黑树确保没有一条路径会比其他路径长出两倍，因此红黑树是一种弱平衡二叉树，相对于要求严格的AVL树来说，它的旋转次数少，所以对于搜索、插入、删除操作较多的情况一般使用红黑树（插入最多两次旋转、删除最多三次旋转）。

红黑树在查找、插入删除的性能都是`O(logn)`，并且性能稳定，所以`STL`里边的很多结构包括`map`的底层实现都是红黑树。

#### B树、B+树

[图文 红黑树,B树,B+树 本质区别及应用场景](https://blog.csdn.net/qq_29373285/article/details/88610654>)

文件很大，不可能全部存储在内存中，故要存储到磁盘上。

b树（balance tree）和b+树应用在数据库索引，可以认为是m叉的多路平衡查找树

#### [哈夫曼编码](https://www.cnblogs.com/-citywall123/p/11297523.html)

哈夫曼编码，主要目的是根据使用频率来最大化节省字符（编码）的存储空间。主要应用在数据压缩，加密解密等场合。

权值大的在上层，权值小的在下层。满足出现频率高的码长短。

哈夫曼编码的带权路径权值：叶子节点的值 \* 叶子节点的高度（根节点为0）。

### 图结构

#### 拓扑排序

对一个[有向无环图](https://baike.baidu.com/item/有向无环图/10972513)(Directed Acyclic Graph简称DAG)G进行拓扑排序，是将G中所有顶点排成一个线性序列，使得图中任意一对顶点u和v，若边<u,v>∈E(G)，则u在线性序列中出现在v之前。通常，这样的线性序列称为满足拓扑次序(Topological Order)的序列，简称拓扑序列。简单的说，由某个集合上的一个[偏序](https://baike.baidu.com/item/偏序/2439087)得到该集合上的一个[全序](https://baike.baidu.com/item/全序/10577699)，这个操作称之为拓扑排序。

- [拓扑排序详解与实现](https://www.cnblogs.com/bigsai/p/11489260.html)
- 应用：判断有向图中是否有环或者回路 -- [leetcode207:课程表](<https://leetcode-cn.com/problems/course-schedule/>)

#### 深度优先遍历

#### 广度优先遍历

### Hash Table

`Hash table`的实现主要包括构造`Hash`和处理`Hash`冲突两个方面

- 构造`Hash`：

  - 直接定址法
  - 除留余数法
  
  - 数字分析法
  - 折叠法
  - 平方取中法
  
- [`Hash`冲突解决](<https://www.cnblogs.com/gongcheng-/p/10894205.html#_label1_4>)：$\textcolor{red}{经过哈希函数处理后有不同的数据对应相同的值。}$

  1.开放定址法
  
  - 线性探测再散列:按顺序决定值时，如果某数据的值已经存在，则在原来值的基础上往后加一个单位，直至不发生哈希冲突
  - 二次探测再散列:按顺序决定值时，如果某数据的值已经存在，则在原来值的基础上先加1的平方个单位，若仍然存在则减1的平方个单位。随之是2的平方，3的平方等等。直至不发生哈希冲突。
  - 伪随机探测再散列:按顺序决定值时，如果某数据已经存在，通过随机函数随机生成一个数，在原来值的基础上加上随机数，直至不发生哈希冲突.
  
  2.再哈希法:对于冲突的哈希值再次进行哈希处理，直至没有哈希冲突
  3.链地址法：对于相同的值，使用链表进行连接。使用数组存储每一个链表。
  4.建立一个公共溢出区:建立公共溢出区存储所有哈希冲突的数据。

## `C++`语法

### `static`作用

- 修饰普通变量，修改变量的存储区域和生命周期，使变量存储在静态区（在整个函数运行期间一直存在），在 main 函数运行前就分配了空间，如果有初始值就用初始值初始化它，如果没有初始值系统用默认值（0）初始化它。
- 修饰普通函数，表明函数的作用范围，仅在定义该函数的文件内才能使用。在多人开发项目时，为了防止与他人命名空间里的函数重名，可以将函数定位为 static。
- 修饰成员变量，修饰成员变量使所有的对象只保存一个该变量，而且不需要生成对象就可以访问该成员。类的静态成员为其所有对象共享，不管有多少对象，静态成员只有一份存于公用内存中。
- 修饰成员函数，修饰成员函数使得不需要生成对象就可以访问该函数，但是在 static 函数内不能访问非静态成员。

### `const`的使用

- 限定变量为不可修改

- 限定成员函数不可以修改任何数据成员

- const与指针

  - const char *p 表示 指向的内容不能改变

  - char * const p，就是将P声明为常指针，它的地址不能改变，是固定的，但是它的内容可以改变

  - [const char * 、char const *、 char * const 三者的区别](https://blog.csdn.net/qq_40244176/article/details/80765975)

    ```c++
    const char *ptr; // 指向字符常量的指针。ptr是一个指向 char* 类型的常量，所以不能用ptr来修改所指向的内容，换句话说，*ptr的值为const，不能修改。
    char const *ptr; // 此种写法和const char *等价
    char * const ptr; // 一个指向字符的指针常数，即const指针
    ```

### c++四种cast转化

<<<<<<< HEAD
![图片](Pictures\C++\cast强制类型转换.png?raw=true)

### 指针和引用

![图片](Pictures/C++/指针和引用.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\C++\cast强制类型转换.png?raw=true)

### 指针和引用

![](https://github.com/ren2504413601/2020interview/tree/main/Pictures/C++/指针和引用.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

### 源码到可执行文件的过程

预编译  --> 编译 --> 汇编 --> 链接



预处理阶段：处理以 `#` 开头的预处理命令； 

编译阶段：翻译成汇编文件； 

汇编阶段：将汇编文件翻译成可重定位目标文件； 

链接阶段：将可重定位目标文件和 `printf.o` 等单独预编译好的目标文件进行合并，得到最终的可执行目标文件。

### 静态链接和动态链接区别

- 静态链接就是在编译期间，由编译器和连接器将静态库集成到应用程序内，并制作成目标文件以及可以独立运作的可执行文件。静态库一般是一些外部函数与变量的集合。 

  静态库很方便，但是如果我们只是想用库中的某一个函数，却仍然得把所有的内容都链接进去。一个更现代的方法是使用共享库，避免了在文件中静态库的大量重复。 

  动态链接可以在首次载入的时候执行，也可以在程序开始执行的时候完成。这个是由动态链接器完成，比方标准 `C` 库(`libc.so`) 通常就是动态链接的，这样所有的程序可以共享同一个库，而不用分别进行封装。

- 动态库一般都会存在/usr/lib/ 目录下；而静态库可以在任何目录下，只要你第一次链接的时候，用绝对路径去链接就行了，之后再删除，是不会影响你的生成的执行文件的。

  静态链接的过程就已经把要链接的内容已经链接到了生成的可执行文件中，就算你在去把静态库删除也不会影响可执行程序的执行；而动态链接这个过程却没有把内容链接进去，而是在执行的过程中，再去找要链接的内容，生成的可执行文件中并没有要链接的内容，所以当你删除动态库时，可执行程序就不能运行。

  总的来说，动态链接生成的可执行文件要比静态链接生成的文件要小一些。

  静态链接库执行速度比动态链接库快。（执行过程不需要找链接的内容）。动态链接库更节省内存。（未写入要链接的内容）

### `STL`容器和算法

#### 序列容器、关联容器

- 序列容器：`deque`、`vector`、`list`、`forward_list`、`array`、`string`

<<<<<<< HEAD
![图片](Pictures\C++\顺序容器.png?raw=true)

- 关联容器:

![图片](Pictures\C++\关联容器.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\C++\顺序容器.png?raw=true)

- 关联容器:

![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\C++\关联容器.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

1.vector 底层数据结构为数组 ，支持快速随机访问
2.list 底层数据结构为双向链表，支持快速增删
3.deque 底层数据结构为一个中央控制器和多个缓冲区，详细见STL源码剖析P146，支持首尾（中间不能）快速增删，也支持随机访问
deque是一个双端队列(double-ended queue)，也是在堆中保存内容的.它的保存形式如下:
[堆1] --> [堆2] -->[堆3] --> ...
每个堆保存好几个元素,然后堆和堆之间有指针指向,看起来像是list和vector的结合品.
4.stack 底层一般用list或deque实现，封闭头部即可，不用vector的原因应该是容量大小有限制，扩容耗时
5.queue 底层一般用list或deque实现，封闭头部即可，不用vector的原因应该是容量大小有限制，扩容耗时
（stack和queue其实是适配器,而不叫容器，因为是对容器的再封装）
6.priority_queue 的底层数据结构一般为vector为底层容器，堆heap为处理规则来管理底层容器实现
7.set 底层数据结构为红黑树，有序，不重复
8.multiset 底层数据结构为红黑树，有序，可重复 
9.map 底层数据结构为红黑树，有序，不重复
10.multimap 底层数据结构为红黑树，有序，可重复
11.hash_set 底层数据结构为hash表，无序，不重复
12.hash_multiset 底层数据结构为hash表，无序，可重复 
13.hash_map 底层数据结构为hash表，无序，不重复
14.hash_multimap 底层数据结构为hash表，无序，可重复

#### `STL`迭代器删除元素

<<<<<<< HEAD
![图片](Pictures\C++\STL迭代器删除元素.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\C++\STL迭代器删除元素.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

#### `map`和`unordered_map`比较

- `map`，底层基于红黑树。
  - 优点：有序的
  - `map`查找、删除等操作时间复杂度稳定，都是`O(logn)`

- `unordered_map`，底层基于哈希表
  - 优点：查找、插入、删除等操作平均时间复杂度是`O(c)`
  - 缺点：基于哈希表，内部存储以`(key,value)`方式，空间占用率高。`unorderd_map`查找、插入、删除等操作时间复杂度不稳定，平均是`O(c)`，极端情形会是`O(n)`，这取决于哈希函数。

#### 数组和链表的区别

- 数组是将元素在内存中连续存放，每个元素的占用空间一样，可以通过下标迅速访问数组中的元素。但是数组的插入和删除操作效率很低。插入数据时，这个位置后边的数据都要后移。删除时，这个位置后边的数据都要前移。
- 链表的元素在内存中不是顺序存储的，而是通过元素间的指针联系到一块。比如，上一个元素有一个指针指向下一个元素，以此类推，直到最后一个元素。访问链表的元素需要从链表头元素开始直到找到这个元素。但是插入和删除操作对于链表来说相对简单。

### [`C++ `内存管理](https://www.cnblogs.com/qiubole/archive/2008/03/07/1094770.html)

#### 堆、栈、静态内存、栈内存、堆上的动态对象

变量类别：

根据作用域可分为全局变量和局部变量。

根据生命周期可分为静态存储方式和动态存储方式。

- 全局对象：在程序启动时分配，程序结束时销毁。
- 局部对象：进入其定义所在的程序块时被创建，在离开时销毁。
- 局部`static`对象：在第一次使用前分配，程序结束时销毁。



全局变量、静态变量存储区：用来保存局部`static`对象、类`static`数据成员以及定义在任意函数之外的变量（全局变量）

常量数据区：保存字面值常量。

栈内存：保存定义在函数内的非`static`对象

自由空间或者堆：动态对象的生命周期由程序控制。这也就是说，当动态对象不再被使用时，代码必须显式地销毁。

**常量**在C++里的定义就是一个top-level const 加上对象类型，常量的定义必须初始化。对于局部对象，常量存放在栈区，对于全局对象，常量存放在全局/静态存储区。对于字面值常量，常量存放在常量存储区。

#### [堆与栈的区别](https://blog.csdn.net/K346K346/article/details/80849966/)

程序内存分配场景，两种内存分配方式

数据结构场景，两种常用数据结构

比较：

- 栈由操作系统自动分配释放 ，用于存放函数的参数值、局部变量等，其操作方式类似于数据结构中的栈。堆由开发人员分配和释放， 若开发人员不释放，程序结束时由 OS 回收，分配方式类似于链表。

#### inline函数

在 c/c++ 中，为了解决一些频繁调用的小函数大量消耗栈空间（栈内存）的问题，特别的引入了 inline 修饰符，表示为内联函数。

在系统下，栈空间是有限的，假如频繁大量的使用就会造成因栈空间不足而导致程序出错的问题，如函数的死循环递归调用的最终结果就是导致栈内存空间枯竭。

建议 inline 函数的定义放在头文件中

- 相当于把内联函数里面的内容写在调用内联函数处；
- 相当于不用执行进入函数的步骤，直接执行函数体；

#### `new` 和 `delete`,`malooc`和`free`

<<<<<<< HEAD
![图片](Pictures\C++\new,malloc.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\C++\new,malloc.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

- `malloc` 和 `free`

```c++
#include <stdlib.h>
// malloc free 声明
void *malloc(size_t size);
void free(void *ptr);
```

```c++
char *str;
// malloc
str = (char *) malloc(15);
// free
free(str);

int * ptr = (int *) malloc( sizeof(int) * 10);//分配一个10个int元素的数组
free(ptr);
```

- `new`、`delete`

```c++
//开辟单地址空间
int *p = new int;  //开辟大小为sizeof(int)空间
int *q = new int(5); //开辟大小为sizeof(int)的空间，并初始化为5。
//开辟数组空间
//一维
int *a = new int[100]{0};//开辟大小为100的整型数组空间，并初始化为0。
//二维
int (*a)[6] = new int[5][6];
//三维
int (*a)[5][6] = new int[3][5][6];
//四维及以上以此类推。
```

```c++
//释放单个int空间
int *a = new int;
delete a;
//释放int数组空间
int *b = new int[5];
delete []b;
```

  

  `new`内存分配失败时，会抛出`bac_alloc`异常，它不会返回NULL；`malloc`分配内存失败时返回`NULL`

#### 左值、右值

[左值和右值](https://blog.csdn.net/weixin_43869906/article/details/89429268)

[右值引用&&](https://www.cnblogs.com/xiangtingshen/p/10366697.html)

左值通常可以取地址，有名字的值就是左值。不能取地址，没有名字的值就是右值。

左值是对应内存中有确定存储地址的对象的表达式的值，而右值是所有不是左值的表达式的值。

一般来说，左值是可以放到赋值符号左边的变量。但能否被赋值不是区分左值与右值的依据。比如，C++的const左值是不可赋值的；而作为临时对象的右值可能允许被赋值。

左值与右值的**根本区别在于是否允许取地址&运算符获得对应的内存地址。**

在c++11中，右值由两个概念构成，将亡值和纯右值。纯右值是用于识别临时变量和一些不与变量关联的值，如：1+3产生的临时变量，2，true等，将亡值是指具有转移语义的对象，比如返回右值引用T&&的函数返回值等。

右值引用：

```c++
T&& a = ReturnRvalue();
// 假设RetureRvalue()函数返回一个右值，那么上述语句声明了一个名为a的右值引用
// 其值等于ReturnRvalue()函数返回的临时变量的值
```

基于右值引用可以实现转移语义和完美转发新特性。

左值引用和右值引用的区别：

- 左值可以寻址、右值不可以
- 左值可以被赋值，右值不可以被赋值，右值可以用来给左值赋值
- 左值可变、右值不可变（仅对基础类型适用，用户自定义类型右值引用可以通过成员函数改变）

#### 智能指针

智能指针是一个`RAII（Resource Acquisition is initialization）`类模型，用来动态的分配内存，解决申请的空间在函数结束时忘记释放而造成内存泄漏的问题。使用智能指针可以很大程度上解决这一问题，因为智能指针就是一个类，当超出了时，类会自动调用析构函数，析构函数会自动释放资源。所以智能指针的作用原理就是在函数结束时自动释放内存空间，不需要手动释放内存空间。

普通指针与内存释放

```c++
void Foo( )
{ 
    int* iPtr = new int[5];  
    //manipulate the memory block . . .  
    delete[ ] iPtr;
 }
```

理想状态下，上述程序运行的很好，内存也可以释放回去。但是有些异常情况：如访问无效的内存地址，除法分母为0，或者另外的程序员修改了一个Bug（根据一个条件添加了过早的返回语句）。

[C++11 智能指针](https://www.jianshu.com/p/e4919f1c3a28)

- auto_ptr(c++98的方案，cpp11已经抛弃)

  采用所有权模式。存在的问题：

  - 当把一个`auto_ptr`赋给另外一个`auto_ptr`时，它的所有权(ownship)也转移了
  - auto_ptr不能指向一组对象，就是说它不能和操作符new[]一起使用。
  - auto_ptr不能和STL标准容器（vector、list、map）等一起使用

- unique_ptr

  `unique_ptr`也是对`auto_ptr`的替换。`unique_ptr`遵循着独占语义。在任何时间点，资源只能唯一地被一个`unique_ptr`占有。当`unique_ptr`离开作用域，所包含的资源被释放。如果资源被其它资源重写了，之前拥有的资源将被释放。所以它保证了他所关联的资源总是能被释放。

  ```c++
  // 创建一个 unique_ptr
  unique_ptr<int> uptr( new int );
  // unique_ptr提供了创建数组对象的特殊方法，当指针离开作用域时，调用delete[]代替delete
  unique_ptr<int[ ]> uptr( new int[5] );
  ```

  当程序unique_ptr试图将一个unique_ptr赋值给另一个时，如果源unique_ptr是一个临时右值，编译器允许；如果unique_ptr存在了一段时间，编译器会禁止。

  ```c++
  int main()
  {
      unique_ptr<string> up1(new string("Hello world"));
      unique_ptr<string> up2;
      up2 = up1; // #1 不能赋值，会报错
  
      unique_ptr<string> up3;
      up3 = unique_ptr<string>(new string("Hello")); // # 允许赋值，因为此时赋值的对象是一个右值临时的对象
      
      // c++ 标准库函数 std::move() 实现了将一个 unique_ptr 赋值给另一个
      unique_ptr<string> ps1, ps2;
      ps1 = unique_ptr<string>(new string("Hello"));
      ps2 = move(ps1);
      ps1 = unique_ptr<string>(new string("ABC"));
      cout << *ps1 << " " << *ps2;  // -> ABC Hello
  }
  ```

- shared_ptr

  共享所有权。多个指针可以同时指向一个对象，当最后一个shared_ptr离开作用域时，内存才会自动释放。采用计数机制来表明资源被几个指针共享。可以通过成员函数use_count()来查看资源的所有者个数。

  ```c++
  void main( )
  {
      // 创建
  	shared_ptr<int> sptr1( new int );
      // 使用 make_shared 宏来加速创建的过程
      shared_ptr<int> sptr1 = make_shared<int>(100);
  }
  ```

- weak_ptr

  weak_ptr是一种不控制对象生命周期的智能指针，它指向一个shared_ptr管理的对象。

  进行该对象的内存管理的是那个强引用的shared_ptr，weak_ptr只是提供了对管理对象的一个访问手段。

  weak_ptr设计的目的是为配合shared_ptr而引入的一种智能指针来协助shared_ptr工作，可以从一个shared_ptr或者另一个weak_ptr对象构造，其构造与析构不会引起引用计数的增加或者减少。可以用来解决shared_ptr相互引用造成的死锁问题。

  ```c++
  void main( )
  {
      shared_ptr<Test> sptr( new Test );
  	// 从 shared_ptr构造一个weak_ptr
      weak_ptr<Test> wptr( sptr );
      // 从 weak_ptr调用lock()可以得到shared_ptr或者直接将类型转换为 shared_ptr
      shared_ptr<Test> sptr2 = wptr.lock( );
  }
  ```

  

### C++面向对象

#### `sizeof()`

- 对数组，得到整个数组所占空间大小

- 对指针，得到指针本身所占空间大小

- `memset(next, 0, sizeof(next));`

  ```c++
  // memset declartion
  void* memset( void* dest, int ch, std::size_t count );
  /**
   * 参数
   * 	dest	-	指向要填充的对象的指针
   * 	ch	-	填充字节
   * 	count	-	要填充的字节数
   * 返回值
   * 	dest
  */
  
  int arr[3][3];
  memset(arr, -1, sizeof(arr));
  cout << sizeof(arr) << endl; // 36 (9 * 4)
  cout << sizeof(arr[0]) << endl; // 12 (3 * 4)
  cout << sizeof(arr[0][0]) << endl; // 4
  cout << sizeof(int) << endl; // 4
  ```



#### 内存对齐规则

主要是为了性能和平台移植等因素，编译器对数据结构进行了内存对齐

[内存对齐规则之我见](https://levphy.github.io/2017/03/23/memory-alignment.html)

三大规则

1. 对于结构体的各个成员，第一个成员的偏移量是0，排列在后面的成员其当前偏移量必须是当前成员类型的整数倍;
2. 结构体内所有数据成员各自内存对齐后，结构体本身还要进行一次内存对齐，保证整个结构体占用内存大小是结构体内最大数据成员的最小整数倍;
3. 如程序中有#pragma pack(n)预编译指令，则所有成员对齐以n字节为准(即偏移量是n的整数倍)，不再考虑当前类型以及最大结构体内类型。

```c++
#include<iostream>
using namespace std;
struct A{
    char a; 
    int b; 
    short c; 
};
/**
数据成员-	-前面偏移量-	- 成员自身占用
(char) a	0	1
缓冲补齐	1	3(规则1)
(int) b	4	4
(short) c	8	2
缓冲补齐	10	2(规则2)
*/
struct B{
    short c;
    char a;
    int b;
};
/**
数据成员-	-前面偏移量-	-成员自身占用
short c	0	2
char a	2	1
缓冲补齐	3	1(规则1)
int b	4	4
*/
int main(){
    cout << sizeof(A) << endl; // 12
    cout << sizeof(B) << endl; // 8
    return 0;
}
```



#### `struct`和`class`

总的来说，`struct`更适合看作一种数据结构的实现体，`class`更适合看作一个对象的实现体

C++为了兼容C，保留了struct关键字，但是实际上C++中的struct是一个默认访问控制权限为public的class。C++标准规定:一个空类的大小为1个字节，因此在C++中，sizeof(空类或空结构体) = 1，在C语言中，sizeof(空结构体) = 0。

区别： 

- 默认的继承访问权限。`struct`是`public`的，`class`是`private`的
- `struct` 作为数据结构的实现体，它默认的数据访问控制是 `public` 的，而 `class `作为对象的实现体，它默认的成员变量访问控制是` private` 的。

#### 成员初始化列表

好处

- 更高效：少了一次调用默认构造函数的过程。
- 有些场合必须要用初始化列表：
  1. 常量成员(const成员)，因为常量只能初始化不能赋值，所以必须放在初始化列表里面
  2. 引用类型（引用成员），引用必须在定义的时候初始化，并且不能重新赋值，所以也要写在初始化列表里面
  3. 没有默认构造函数的类类型，因为使用初始化列表可以不必调用默认构造函数来初始化
  4. 如果类存在继承关系，派生类必须在其初始化列表中调用基类的构造函数

#### [C++ 为什么拷贝构造函数参数必须为引用？赋值构造函数参数也必须为引用吗？](https://www.cnblogs.com/chengkeke/p/5417362.html)

拷贝构造函数的参数必须为引用。赋值构造函数参数既可以为引用，也可以为值传递，值传递会多一次拷贝。因此建议赋值构造函数建议也写为引用类型。

#### this 指针

C++ 中，每一个对象都能通过 **this** 指针来访问自己的地址。**this** 指针是所有成员函数的隐含参数。因此，在成员函数内部，它可以用来指向调用对象。

在以下场景中，经常需要显式引用`this`指针：

1. 为实现对象的链式引用；
2. 为避免对同一对象进行赋值操作；
3. 在实现一些数据结构时，如 `list`

#### public、protected及private用法

- 用户代码（类外）可以访问public成员而不能访问private成员；private成员只能由类成员（类内）和友元访问。$\textcolor{red}{封装的体现}$
- protected成员可以被派生类对象访问，不能被用户代码（类外）访问。$\textcolor{red}{继承的体现}$
- 继承包括三种继承方式：
  - public继承：基类public成员，protected成员，private成员的访问属性在派生类中分别变成：public, protected, private
  - protected继承：基类public成员，protected成员，private成员的访问属性在派生类中分别变成：protected, protected, private
  - private继承：基类public成员，protected成员，private成员的访问属性在派生类中分别变成：private, private, private

#### friend 友元类和友元函数

- 能访问私有成员
- 破坏封装性
- 友元关系不可传递
- 友元关系的单向性
- 友元声明的形式及数量不受限制

#### 子类构造与析构时，父类构造与析构机制

- 构造子类对象时，先调用父类构造函数，再调用子类构造函数（构造函数没有虚函数这一说法）
- 析构子类对象时，先调用子类的析构函数，再调用父类析构函数（无论父类的析构函数是否为虚函数）
- 构造子类构造的父类对象时，先调用父类构造函数，再调用子类构造对象（构造函数没有虚函数这一说）
- 析构子类构造的父类对象时：

1. 若父类是虚函数，则先调用子类析构函数，再调用父类析构函数
2. 若父类不是虚函数，则只调用父类的析构函数

#### [c++虚函数](https://www.zhihu.com/question/23971699)

- 虚函数、纯虚函数

  `virtual`关键字。声明方式例如 `virtual void foo()`，`virtual void funtion1()=0`

  - 定义一个函数为虚函数，不代表函数为不被实现的函数。定义他为虚函数是为了允许用基类的指针来调用子类的这个函数。

  - 定义一个函数为纯虚函数，才代表函数没有被实现。定义纯虚函数是为了实现一个接口，起到一个规范的作用，规范继承这个类的程序员必须实现这个函数。称带有纯虚函数的类为抽象类。

  - [构造函数不能声明为虚函数，析构函数要声明为虚函数](https://www.cnblogs.com/wuyepeng/p/9882289.html)

    创建一个对象必须明确指出它的类型，否则无法创建，一个对象创建成功编译器获得它的实际类型，然后去调用对应的函数，而如果构造函数声明为虚函数，会形成一个死锁，虚函数是在运行才能确定确定其调用哪一个类型的函数，而具体哪一个类型是编译器通过对象的类型去确定的，但是此时对象还未创建也就没法知道其真实类型。

    虚函数和虚函数表是两个不同的东西，虚函数的调用是在运行时才确定的，虚函数表是在编译时就已经确定的了 。

  [C++中基类的析构函数不是虚函数，会带来什么问题!!](<https://blog.csdn.net/weixin_30389003/article/details/96294206>)
  
  - 基类的的析构函数不是虚函数的话，删除指针时，只有其类的内存被释放，派生类的没有。这样就内存泄漏了。
  - 析构函数不是虚函数的话，直接按指针类型调用该类型的析构函数代码，因为指针类型是基类，所以直接调用基类析构函数代码。
  - 当基类指针指向派生类的时候，如果析构函数不声明为虚函数，在析构的时候，不会调用派生类的析构函数，从而导致内存泄露。
  
  为什么**C++默认的析构函数不是虚函数**
  
  - C++默认的析构函数不是虚函数是因为虚函数需要额外的虚函数表和虚表指针，会占用额外的内存。对于不会继承的类来说，析构函数是虚函数会造成内存的浪费。因此C++默认的析构函数不是虚函数，但是作为父类时要设置为虚函数。
  
  前缀树的实现及其运用

```c++
#include<bits/stdc++.h>
using namespace std;

/**
 * LC208. 实现 Trie (前缀树)
 * 实现一个 Trie (前缀树)，包含 insert, search, 和 startsWith 这三个操作
*/
class LC208 {
private:
    bool isEnd;
    Trie* next[26];
public:
    /** Initialize your data structure here. */
    Trie() {
        isEnd = false;
        memset(next, 0, sizeof(next));
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        Trie* node = this;
        for (char& c : word)
        {
            if (node -> next[c - 'a'] == NULL)
            {
                node -> next[c - 'a'] = new Trie();
            }
            node = node -> next[c - 'a'];
        }
        node -> isEnd = true;
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        Trie* node = this;
        for (char& c : word)
        {
            if (node -> next[c - 'a'] == NULL)
            {
                return false;
            }
            node = node -> next[c - 'a'];
        }
        return node -> isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        Trie* node = this;
        for (char& c : prefix)
        {
            if (node -> next[c - 'a'] == NULL)
            {
                return false;
            }
            node = node -> next[c - 'a'];
        }
        return true;
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */

/**
 * LC212. 单词搜索 II
*/

class Trie
{
    public:
    Trie* next[26];
    string words;
    Trie()
    {
        words = "";
        memset(next, 0, sizeof(next));
    }
};
class LC212 {
public:
    void dfsHelper(vector<vector<char>>& board, Trie* node, int x, int y, vector<string>& ans)
    {
        int m = board.size(), n = board[0].size();
        char tmpc = board[x][y]; 
        if (tmpc == '.' || node -> next[tmpc - 'a'] == NULL) return;

        node = node -> next[tmpc - 'a'];

        if (node -> words != "")
        {
            ans.push_back(node ->words);
            node ->words = ""; // 去重， 很重要
        }

        board[x][y] = '.';
        if (x > 0) dfsHelper(board, node, x - 1, y, ans);
        if (x + 1 < m) dfsHelper(board, node, x + 1, y, ans);
        if (y > 0) dfsHelper(board, node, x, y - 1, ans);
        if (y + 1 < n) dfsHelper(board, node, x, y + 1, ans);
        board[x][y] = tmpc;
    }
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) 
    {
        vector<string> ans;
        int m = board.size();
        if (m == 0) return ans;
        int n = board[0].size();

        // 建 trie 树
        Trie* node = new Trie();
        for (string& s : words)
        {
            Trie* curr = node;
            for (char& c: s)
            {
                if (curr -> next[c - 'a'] == NULL)
                {
                    curr -> next[c - 'a'] = new Trie();
                }
                curr = curr -> next[c - 'a'];
            }
            curr -> words = s;
        }
        // dfs 
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                dfsHelper(board, node, i, j, ans);
            }
        }
        return ans;
    }
};
```

- 这里`trie`树的实现使用了`this`指向类对象的地址，可以实现对类成员函数的访问。
- `Trie* next[26]`在类`LC208`中是private的，在`LC212`是公有的，这里主要区别在于前者是在类内调用，后者则在类外使用

## `Python` 语法

### [`python`内存管理机制](<https://baijiahao.baidu.com/s?id=1625794283727801503&wfr=spider&for=pc>)

`python`中使用引用计数，来保持追踪内存中的对象，`python`内部记录了对象有多少个引用，即引用计数，当对象被创建时就创建了一个引用计数，当对象不再需要时，这个对象的引用计数为0时，被当做垃圾回收。

- 查看对象的引用计数：`sys.getrefcount()`

  ```python
  >>> import sys
  >>> a = [1, 2]
  >>> sys.getrefcount(a)
  2
  >>> b = a
  >>> sys.getrefcount(b)
  3
  >>> sys.getrefcount(a)
  3
  >>> del a
  >>> sys.getrefcount(a)
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  NameError: name 'a' is not defined
  >>> sys.getrefcount(b)
  2
  >>> b
  [1, 2]
  ```

- 引用计数`+1`情形

  1.对象被创建：x=4

  2.另外的别人被创建：y=x

  3.被作为参数传递给函数：foo(x)

  4.作为容器对象的一个元素：a=[1,x,'33']

- 引用计数减少情形

  1.一个本地引用离开了它的作用域。比如上面的foo(x)函数结束时，x指向的对象引用减1。

  2.对象的别名被显式的销毁：del x ；或者del y

  3.对象的一个别名被赋值给其他对象：x=789

  4.对象从一个窗口对象中移除：myList.remove(x)

  5.窗口对象本身被销毁：del myList，或者窗口对象本身离开了作用域。

- 内存池机制
  - `Python`提供了对内存的垃圾收集机制，但是它将不用的内存放到内存池而不是返回给操作系统
  - Python中所有小于256个字节的对象都使用`pymallo`c实现的分配器，而大的对象则使用系统的 `malloc`。另外`Python`对象，如整数，浮点数和`List`，都有其独立的私有内存池，对象间不共享他们的内存池。也就是说如果你分配又释放了大量的整数，用于缓存这些整数的内存就不能再分配给浮点数。

### [is和==的区别](<https://blog.csdn.net/Tonywu2018/article/details/89022355>)

`python`是一种面向对象的语言，`python`中对象包含三种基本要素：`id`(返回的是对象的地址)、`type`(返回的是对象的数据类型)及`value`(对象的值)

`is`比较的是两个对象的地址值，也就是说两个对象是否为同一个实例对象；而`==`比较的是对象的值是否相等，其调用了对象的`__eq__`方法。`a is b` 相当于 $id(a) == id(b)$ 。$a == b$相当于 $a.\_\_eq\_\_(b)$。

$\textcolor{red}{进行None判断时，为什么使用is或者not，而不是用==?}$

`None`在`python `中是一个单例对象，一个变量如果是`None`。它一定和`None`指向同一内存地址。`None`是`python`中的一个特殊的变量，表示一个空的对象。而空值是`python`的一个特殊值，$\color{red}{数据为空并不代表是空对象}$，例如$[],\{\},(),''$等都不是`None`。

在Python中，None、空列表[]、空字典{}、空元组()、0等一系列代表空和无的对象会被转换成False。除此之外的其它对象都会被转化成True。

### [`str`的`join`和`+`的区别](<https://www.cnblogs.com/Sandy-1128/p/python-sandy-0404.html>)

```python
>>> str1 = " ".join(["hello", "world"])
>>> str1
'hello world'
>>> str2 = "hello " + "world"
>>> str2
'hello world'
```

结果一样，但$\color{red}{join的性能明显好于+。这是为什么呢？}$

**原因：**

字符串是不可变对象，当用操作符+连接字符串的时候，每执行一次+都会申请一块新的内存，因此用+连接字符串的时候会涉及好几次内存申请和复制。

而join在连接字符串的时候，会先计算需要多大的内存存放结果，然后一次性申请所需内存并将字符串复制过去，这是为什么join的性能优于+的原因。

所以在连接字符串数组的时候，我们应考虑优先使用join。

### 可变类型和不可变类型

- 不可变：`number`、`string`、`tuple`
- 可变：`dict`、`list`、`set`

### 浅拷贝和深拷贝

- 对象的赋值实质上是对象的引用，不会开辟新空间

- 浅拷贝：`copy`模块里边的`copy()`方法实现。拷贝了最外层容器，副本中的元素是源容器中元素的引用。
  - 对于不可变类型 `Number、String、Tuple`,浅拷贝仅仅是地址指向，不会开辟新空间。
  - 对于可变类型 `List、Dictionary、Set`，浅拷贝会开辟新的空间地址(仅仅是最顶层开辟了新的空间，里层的元素地址还是一样的)，进行浅拷贝。
  - 对于列表和其它可变序列来说，还可以使用简洁的 `l2 = l1[:]`语句创建副本。
- 深拷贝：`copy.deepcopy()`
  - 深拷贝，除了最外层容器的拷贝，还对子元素也进行了拷贝（本质上递归浅拷贝）
  - 经过深拷贝后，原始对象和拷贝对象所有的元素地址都没有相同的了

### `*args和**kwargs`

是`python`中的可变参数。`args`表示任何多个无名参数，它是一个`tuple`；`kwargs`表示关键字参数，它是一个`dict`。并且同时使用`args`和`kwargs`时，必须`args`参数列要在`kwargs`前。

### `sort()`和`sorted()`

```python
# function
sort(*, key=None, reverse=False)
sorted(iterable, cmp=None, key=None, reverse=False)
# iterable -- 可迭代对象
# cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
# key -- 排序用来比较的元素
# resverse -- reverse == True（降序）， reverse == False(升序), 默认升序排列 
```

- `list.sort()` 函数只适用于列表排序，而`sorted()`函数适用于任意可以迭代的对象排序（）。
- `list.sort()`是原地(`in-place`)修改，`sorted()`不是原地修改。使用`list.sort()`可以节省空间，提高效率。

### [装饰器decorate](https://www.zhihu.com/question/26930016)

装饰器是可以调用的对象，本质上是一个Python函数，其参数是另一个函数（被装饰的函数）。它可以让其他函数在不需要做任何代码变动的前提下增加额外功能，装饰器的返回值也是一个函数对象。概括的讲，装饰器的作用就是为已经存在的对象添加额外的功能。

```python
# reference to fluent python chapter 7
# decr()函数返回inner函数对象
>>> def deco(func):
...     def inner():
...             print("running inner()")
...     return inner
# 使用 decr 装饰 target
>>> @deco
... def target():
...     print("running target()")
...
# 调用被装饰的 target 发现其实是会运行 inner
>>> target()
running inner()
# 最后发现 target 是 inner 的引用
>>> target
<function deco.<locals>.inner at 0x00000269AE13B8C8>
```

装饰器的特性：

- 能把装饰的函数替换成其他函数

- 它们在被装饰的函数定义之后立即运行。这通常是在导入时（即`python`加载模块时）。

### 生成器（`generator`）

- 迭代器协议。对象需要提供`next`方法，它要么返回迭代中的下一项，要么就引起一个`StopIteration`异常，终止迭代

- 生成器

  - 生成器函数

    常规函数定义。使用`yeild`语句而不是`return` 返回结果。`yield`语句一次返回一个结果，在每个结果中间，挂起函数的状态，以便下次重它离开的地方继续执行。

    ```python
    # 使用生成器函数
    def genSquare(N):
        for i in range(N):
            yeild i**2
            
    for item in genSquare(5):
        print(item)
        
    # 使用一般函数
    def genSquare(i):
        return i**2
    
    for item in range(5):
        print(genSquare(iteam))
    ```

  - 生成器表达式

    ```python
    # 列表表达式： 
    >>> square = [x**2 for x in range(5)]
    >>> square
    [0, 1, 4, 9, 16]
    
    #生成器表达式：
    >>> square = (x**2 for x in range(5))
    >>> square
    <generator object <genexpr> at 0x000002158A6CB390>
    >>> next(square)
    0
    >>> next(square)
    1
    >>> next(square)
    4
    >>> next(square)
    9
    >>> next(square)
    16
    >>> next(square)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    StopIteration
    ```

`dict`和`list`的区别

`dict`查找速度快，占用内存大。`list`查找速度慢，占用内存小。`dict`不能存储有序集合。`dict`用`{}`表示，`list`用 `[]`表示

`dict`通过`Hash table`实现

**简单实例：**`dict`按照`value`从大到小排序

```pytho
sorted(d.items(), key = lambda x : x[1], reverse = True)
```

## 数学知识补充

### 柯西（Cauchy）不等式

- 二维形式

$$
(a^2 +b^2) \geq (ac+bd)^2,\quad 当且仅当 ad =bc时等号成立
$$

- 向量形式（`Cauchy-Schwart`不等式）
  $$
  |a| \cdot |b| \geq |a \cdot b|,\quad a=(a_1, a_2,...,a_n),b=(b_1, b_2,...,b_n)
  $$

- 在内积空间上的形式
  $$
  |( \alpha, \beta )| \leq |\alpha| |\beta|
  $$

### 切比雪夫不等式

设随机变量X具有数学期望$E(x) = \mu$，方差$D(X) = \sigma^2$，则对任意正数$ \epsilon $,
$$
\begin{align}
P \{ |X - \mu| < \epsilon \} & \leq 1 - \frac{\sigma^2}{\epsilon^2}  \quad or\\
P \{ |X - \mu| \geq \epsilon \} & \geq  \frac{\sigma^2}{\epsilon^2} 
\end{align}
$$

### 最速下降法、牛顿法、拟牛顿法

- 最速下降法

<<<<<<< HEAD
![图片](Pictures\Math\最速下降法.png?raw=true)

- 牛顿法

![图片](Pictures\Math\牛顿法.png?raw=true)
=======
![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\Math\最速下降法.png?raw=true)

- 牛顿法

![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\Math\牛顿法.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

- 拟牛顿法

  牛顿法的收敛速度快，迭代次数少，但是`Hessian`矩阵很稠密时，每次迭代的计算量很大，随着数据规模增大，`Hessian`矩阵也会变大，需要更多的存储空间以及计算量。

  拟牛顿法就是利用目标函数值$f$和一阶导数$g$的信息，构造出函数的曲率近似，而不需明显形成`Hessian`矩阵，收敛速度也很快。

  在逆牛顿法中，用`Hessian`矩阵的逆矩阵代替牛顿法的`Hessian`矩阵，虽然不能保证最优方向，但是`Hessian`逆始终正定，所以算法始终朝着最优化的方向搜索。

  **拟牛顿条件**：

  假设$f: D \to R $ ，在开集$D \subset R^n $上连续二次可微，$f(x)$在$x_{k+1}$附近的二次近似为：
  $$
  \begin{aligned}
  f(x) 		& \thickapprox f(x_{k+1}) + g_{k+1}^T(x - x_{k+1}) +
  \frac{1}{2}(x - x_{k+1})^T G_{k+1} (x - x_{k+1})
  \end{aligned}
  $$
  对上式两边关于$x$求导，得到：
  $$
  g(x) \thickapprox g_{k+1} + G_{k+1}(x-x_{k+1})
  $$
  令$x=x_k, s_k = x_{k+1}-x_k, y_k = g_{k+1} - g_k$，得：

  
  $$
  \begin{aligned}
  G_{k+1}^{-1} y_k & \thickapprox s_k, \\ 
  \end{aligned}
  $$
  记$H_{k+1} = G_{k+1}^{-1}$称作`Hessian`逆，下式满足的关系称为逆牛顿条件或者逆牛顿方程
  $$
  H_{k+1} y_k = s_k
  $$

  - 对称秩一矫正（SR1）
  - DFT矫正
  - BFGS矫正和PSB矫正

### 智能算法

- 遗传算法

<<<<<<< HEAD
  ![图片](Pictures\Math\遗传算法流程.png?raw=true)
=======
  ![](https://github.com/ren2504413601/2020interview/tree/main/Pictures\Math\遗传算法流程.png?raw=true)
>>>>>>> 043cba1cfba06177ca3548decdaecbb321186cb3

### 最大似然估计（MLE）和最大后验概率(MAP)

- `Maximum Likelihood Estimation(MLE)`提供了一种给定观测数据来估计模型参数的方法，所有采样满足独立同分布的假设。$\color{red}{频率学派模型参数估计的常用方法}$

- `Maximum A Posteriori(MAP)`是根据经验数据获得难以观察统计量的点估计，与`MLE`最大的不同是`MAP`融入了要估计量的先验分布在里边，所以`MAP`可以看作规则化的`MLE`

  $\color{red}{贝叶斯派模型参数估计的常用方法}$

**数学形式：**

假设数据$x_1, x_2, \cdots, x_n$是 `i.i.d`的一组抽样，$X = ( x_1, x_2, \cdots, x_n)$。其中`i.i.d`表示`Independent and identical distribution`独立同分布。那么`MLE`对$\theta$的估计如下：
$$
\begin{aligned}
\hat{\theta}_{MLE} & = \arg \max_{\theta} P(X; \theta) \\
				& = \arg \max_{\theta} P(x_1; \theta) P(x_2; \theta) \cdots P(x_n; \theta) \\
				& = \arg \max_{\theta} \log \prod_{i=1}^n P(x_i; \theta) \\
				& = \arg \max_{\theta} \sum_{i=1}^n \log P(x_i; \theta) \\
				&= - \arg \min_{\theta} \sum_{i=1}^n \log P(x_i; \theta)
\end{aligned}
$$
`MAP`对$\theta$的估计：
$$
\begin{aligned}
\hat{\theta}_{MAP} & = \arg \max_{\theta} P(\theta|X) \\
				& = \arg \min_{\theta} - \log P(\theta|X) \\
				& = \arg \min_{\theta} - \log P(X|\theta) - \log P(\theta) + \log P(X)\\
				& = \arg \min_{\theta} - \log P(X|\theta) - \log P(\theta)	
\end{aligned}
$$
第二行到第三行使用了贝叶斯定理，第三行到第四行$P(X)$可以丢掉因为与 $ \theta $ 无关。观察发现`MLE`和`MAP`在优化时的不同就是在于先验项$- \log P(\theta)$ 

### 凸优化和`Hessian`矩阵正定性在梯度下降中的应用

- 凸函数：函数$f(x)$是凸函数当且仅当对定义中任意两点 `x,y`和实数$\lambda \in [0, 1]$总有
  $$
  f(\lambda x + (1-\lambda y))  \le \lambda f(x) + (1-\lambda) f(y)
  $$

- 凸优化问题
  $$
  \begin{aligned}
  \min & f_0(x) \\
  s.t. & f_i(x) \le 0, i = 1,...,m, \\
  	& h_i(x) = 0, i = 1,...,p
  \end{aligned}
  $$
  其中$f_0(x)$是目标函数，$f_i(x)$是不等式约束条件，$h_i(x)$是等式约束条件。

  一个优化问题是凸优化问题。有三个条件：

  - 目标函数是凸的
  - 不等式约束条件是凸的
  - 等式约束条件是仿射的。($Ax=b$)

- 性质：凸问题的极值点一定是最值点。

若矩阵所有特征值均不小于0,则判定为半正定。若矩阵所有特征值均大于0,则判定为正定。

在判断优化算法的可行性时`Hessian`矩阵的正定性起到了很大的作用,若`Hessian`正定,则函数的二阶偏导恒大于0,函数的变化率处于递增状态,在牛顿法等梯度下降的方法中,`Hessian`矩阵的正定性可以很容易的判断函数是否可收敛到局部或全局最优解。

### $R^2$的数学公式、数学含义？和相关系数有什么区别？

- [决定系数](<https://en.wikipedia.org/wiki/Coefficient_of_determination>)（英语：coefficient of determination，记为$R^2$）

$$
R^2(y, \hat{y}) = 1 - \frac{\sum_{i = 1}^n y_i - \hat{y_i}}{\sum_{i = 1}^n y_i - \bar{y}}
$$

![图片](XinLangWeiBo\RSquare.png)

$R^2$ 是一个评价拟合好坏的指标。这里的拟合可以是线性的，也可以是非线性的。即使线性的也不一定要用最小二乘法来拟合。

- [相关系数]([https://zh.wikipedia.org/wiki/%E7%9A%AE%E5%B0%94%E9%80%8A%E7%A7%AF%E7%9F%A9%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0](https://zh.wikipedia.org/wiki/皮尔逊积矩相关系数)) 用于度量两个变量X和Y之间的[相关](https://zh.wikipedia.org/wiki/相关)程度（线性相关），其值介于-1与1之间.
  $$
  \rho_{X, Y} = \frac{cov(X, Y) }{\sigma_X \sigma_Y} 
  			= \frac{E \left[ (X - \mu_X) (Y - \mu_Y) \right]}{\sigma_X \sigma_Y}
  $$

- 相关系数和$R^2$的关系：[相关系数和R方的关系是什么？](https://www.zhihu.com/question/32021302?sort=created)

  - 对于线性回归的最小二乘拟合，有

  $$
  \rho(x, y) = \pm \sqrt{R^2}
  $$

  ​	因此，就是常说的相关系数表征的是线性关系强弱。而非线性关系时，上式不成立。

  - 对于二维散点进行任意函数的最小二乘拟合，都可以得到
    $$
    \rho(x, y) =  \sqrt{R^2} r
    $$

## 场景题目

### TopK问题

[海量数据中找出前k大数（topk问题）](https://blog.csdn.net/djrm11/article/details/87924616)

Leetcode[215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

- 建堆法。小顶堆（优先队列维护大小为k的最小堆）
- 快速排序+选择

[10亿int整型数，以及一台可用内存为1GB的机器，时间复杂度要求O(n)，统计只出现一次的数](https://blog.csdn.net/qq_35290785/article/details/98672144)

- [位图法（Bitmap）](https://blog.csdn.net/pipisorry/article/details/62443757)
- 可进行数据的快速查找，判重，删除，一般来说数据范围是int的10倍以下
  
- 去重数据而达到压缩数据
- 分治法（哈希分桶、归并排序）

### 【赛马问题】

64匹马，8个跑道，问最少比赛多少场，可以选出跑得最快的4匹马

- Assumptions：每场比赛每个跑道只允许一匹马，且不存在并列情形

- [腾讯算法面试——赛马问题](https://zhuanlan.zhihu.com/p/103572219)

## 面经

- [超详细！腾讯NLP算法岗面经](<https://mp.weixin.qq.com/s/AvK76Kx26mWFa1HJpoua_w>)

  客户端

  <https://www.nowcoder.com/discuss/511515?type=post&order=time&pos=&page=1&channel=1009&source_id=search_post>

  <https://www.nowcoder.com/discuss/418992?type=post&order=time&pos=&page=1&channel=1009&source_id=search_post>

  <https://www.nowcoder.com/discuss/517588?type=2&order=0&pos=66&page=1&source_id=discuss_tag&channel=1009>

  <https://www.nowcoder.com/discuss/tag/732?type=2&order=0>

  <https://www.nowcoder.com/discuss/418992?type=post&order=time&pos=&page=1&channel=1009&source_id=search_post>


