- [树提升方法](#%e6%a0%91%e6%8f%90%e5%8d%87%e6%96%b9%e6%b3%95)
  - [Xgboost](#xgboost)
    - [TREE BOOSTING IN A NUTSHELL](#tree-boosting-in-a-nutshell)
    - [SPLIT FINDING ALGORITHMS](#split-finding-algorithms)
# 树提升方法

## Xgboost

只依赖于每个数据点的在目标函数上的一阶和二阶导数

###   TREE BOOSTING IN A NUTSHELL

改善了回归目标
1. 正则化的学习目标(Regularized Learning Objective)

    对于有n个样本，m个特征的数据集合 $\mathcal{D} = \left \{ \left (  \textbf{x}_i,y_i \right ) \right \} \left ( \left | \mathcal{D}\right | = n,  \textbf{x}_i\in{\mathbb{R}^m}, y_i\in\mathbb{R}  \right )$ ，树提升方法由 $K$ 个输出相加预测目标，即：
    $$y_i =\phi\left ( \textbf{x}_i \right ) =  \sum_{k=1}^{k} f_k\left ( \textbf{x}_i \right ), f_k\in\mathcal{F}$$

    其中 $\mathcal{F}=\left \{  f\left ( \textbf{x} \right )  = \omega_{q\left( \textbf{x} \right)} \right \}\left ( q: {\mathbb{R}^m}  \rightarrow T,  \omega\in{\mathbb{R}^T} \right  )$
      * 上式为回归树(CART)的函数空间
      * $q$为树结构：将每个样本映射到叶子节点
      * $T$为叶子节点个数
      * 每一个$f_k$对应一个树结构q和对应的叶子节点权重$w$
      * 每棵回归树的每一叶子节点上都有一个连续的分数

   从树中取出分裂规则(由q给出)，将样本映射到叶子节点上，从叶子节点中取出权重$w$，将所有权重相加得到最终预测值。
   ![](/images/xgb_figure1.png "打分示例图")

   为了学习模型用到的这个函数集合$\mathcal{F}$，我们最小化以下正则化的目标函数
   $$\mathcal{L}\left ( \phi \right ) = \sum_{i}l\left ( \hat{y}_i, y_i \right )+\sum_{k}\Omega\left ( f_k \right ) $$
   $\Omega\left ( f \right )$为正则函数
   $$\Omega\left ( f \right ) = \gamma T + \frac{1}{2}\lambda\left \| \omega  \right \|^2$$
   * l 为可微分的凸函数，衡量预测值和实际值的残差
   * 增加的正则化项有助于平滑最终学习的权重，避免过拟合

2. 梯度树提升(Gradient Tree Boosting)

    由于$\mathcal{L}\left ( \phi \right )$使用函数作为参数且不能使用传统方法在在欧氏空间优化，使用additive manner训练。$\hat{y}^{\left( t \right)}_i=f_t\left( \textbf{x}_i \right)$为第$i$个样本在第$t$轮的预测值，将$f_t$加入优化目标
    $$\mathcal{L}^{\left ( t \right )} =  \sum_{i=1}^{n} l\left( y_i, \hat{y}^{\left({t-1}\right)}+f\left(\textbf{x}_i \right) \right) + \Omega\left( f_t\right)$$
    贪婪地加入提升最大的$f_t$到模型中(使$\mathcal{L}\left ( \phi \right )$降低最大)，泰勒二阶近似可以很好地优化这个目标(找到$f_t$)
    $$\mathcal{L}^{\left ( t \right )} \simeq \sum_{i=1}^{n} \left [ l\left( y_i, \hat{y}^{\left({t-1}\right)} \right)  + g_if_t \left( \textbf{x}_i\right) + \frac{1}{2} h_i f_t^2 \left( \textbf{x}_i\right) \right  ]$$
    $$g_i = \partial_{\hat{y}^{\left( t-1 \right)} } l\left(\ y_i, \hat{y}^{\left( t-1 \right)}\right)$$
    $$f_i = \partial^2_{\hat{y}^{\left( t-1 \right)} } l\left(\ y_i, \hat{y}^{\left( t-1 \right)}\right)$$
    去掉在$t$步时的常数项，得到最终优化函数
    $$\tilde{\mathcal{L}}^{\left ( t \right )} = \sum_{i=1}^{n} \left [ g_if_t \left( \textbf{x}_i\right) + \frac{1}{2} h_i f_t^2 \left( \textbf{x}_i\right) \right  ] + \Omega\left( f_t\right)$$
    定义落在叶子$j$的样本集合$\textbf{I}_j =\left\{ i | q \left( x_i \right) = j \right\}$，将上式的$\Omega$展开
    $$\tilde{\mathcal{L}}^{\left ( t \right )} = \sum_{i=1}^{n} \left [ g_if_t \left( \textbf{x}_i\right) + \frac{1}{2} h_i f_t^2 \left( \textbf{x}_i\right) \right  ] +  \gamma T + \frac{1}{2}\lambda\left \| \omega  \right \|^2$$
    **计算方法为二次函数极值**，即$ax^2+bx+c$，极值点为$x=-\frac{b}{2a}$，极值为$c-\frac{b^2}{4a}$,对于固定的树结构$q\left( \textbf{x} \right)$。对应$j$计算出最优打分$w^{*}_{j}$的取值为:
    $$w^{*}_{j} = -\frac{\sum_{i\in{I_j}}g_i}{\sum_{i\in{I_j}}h_i+\lambda}$$
    对所有叶子节点，对应的最优打分为(外面加一个sum)：
    $$\tilde{\mathcal{L}}^{\left( t \right)} \left( q\right ) = -\frac{1}{2} \sum^{T}_{j=1} \frac{\left( \sum_{i\in{I_j}}g_i \right)^2}{\sum_{i\in{I_j}}h_i+\lambda} + \gamma T$$
    就像使用不纯度衡量决策树一样，上式可以作为评价一个确定树结构$q$打分质量的评价函数，和不纯度相比不同的的目标函数都可以推导出上式。
    ![](/images/xgb_figure2.png "结构分图")

    通常不可能枚举所有可能的树结构。取而代之的是一个贪婪算法，它从一个节点开始，迭代地向树中添加分支。假设分裂后的左右节点的样本集合是$I_L$和$I_R$，原始集合是$I$，本次损失降低的值可以表达为：
    $$\mathcal{L}_{split} = \frac{1}{2} \left[ \frac{\left( \sum_{i\in{I_L}}g_i \right)^2}{\sum_{i\in{I_L}}h_i+\lambda} + \frac{\left( \sum_{i\in{I_R}}g_i \right)^2}{\sum_{i\in{I_R}}h_i+\lambda} -\frac{\left( \sum_{i\in{I}}g_i \right)^2}{\sum_{i\in{i}}h_i+\lambda} \right] - \lambda$$
    这个公式通常在实践中用于评估候选分裂。

3. 树缩减和列采样(Shrinkage and Column Subsampling)

    * Shrinkage是指每一棵新增的树都乘以一个缩减系数$\eta$
    * 列采样和RF中一样不使用所有特征

    防止过拟合和提升并行计算的速度


### SPLIT FINDING ALGORITHMS

1. Basic Exact Greedy Algorithm

    树学习的核心问题是寻找$\mathcal{L}_{split}$最佳的分裂方式。一种分裂方式是在左右特征中枚举所有可能的样本分裂。对于连续特征要枚举所有的值，因此对于连续特征排序是非常必要的。
  ![](/images/xgb_split1.png "分裂算法1")
2. Approximate Algorithm

    精确贪心算法非常强大，因为它会贪心地枚举所有可能的分裂点。 但是，当数据不能完全放入内存时，不可能有效地执行此操作。
3. Weighted Quantile Sketch
4. Sparsity-aware Split Finding

