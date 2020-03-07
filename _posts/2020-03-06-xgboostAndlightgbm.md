# 树提升方法

## Xgboost

###   TREE BOOSTING IN A NUTSHELL

改善了回归目标
1. 正则化的学习目标
  
    对于有n个样本，m个特征的数据集合$\mathcal{D} = \left \{ \left (  \textbf{x}_i,y_i \right ) \right \} \left ( \left | \mathcal{D}\right | = n,  \textbf{x}_i\in{\mathbb{R}^m}, y_i\in\mathbb{R}  \right )$，树提升方法由$K$个输出相加预测目标，即：
    $$y_i =\phi\left ( \textbf{x}_i \right ) =  \sum_{k=1}^{k} f_k\left ( \textbf{x}_i \right ), f_k\in\mathcal{F}$$

    其中$\mathcal{F}=\left \{  f\left ( \textbf{x} \right )  = \omega_{q\left( \textbf{x} \right)} \right \}\left ( q: {\mathbb{R}^m}  \rightarrow T,  \omega\in{\mathbb{R}^T} \right  )$
      * 上式为回归树(CART)的函数空间
      * q为树结构：将每个样本映射到叶子节点
      * T为叶子节点个数
      * 每一个$f_k$对应一个树结构q和对应的叶子节点权重$w$