# NeuralNetworks And DeepLearning

## A. fundamental knowledge（Introduction）

1. ReLU函数:（修正线性单元）修正->取不小于0的值
2. 序列数据:时间一维
3. 图像数据：RNN（卷积神经网络）
4. ![image-20230116093512817](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230116093512817.png)
   1. SNN：标准神经网络
   2. CNN：卷积神经网络（图像数据）
   3. RNN：循环神经网络（序列数据）
5. 应用领域->supervised Learning
   1. 结构化数据（数据库类型存储数据）|| 短期价值基于结构化数据
   2. 非结构化数据（音频、影像、文字）相较于结构化数据会更加难让机器理解（但对人并非如此）
6. 性能高所要满足的条件（规模推动神经网络的发展）
   1. （computation）规模足够大的神经网络模型
   2. （data）足够多的数据，数据不够多的时候一般不太能够体现出不同神经网络模型的效果差异
   3. （algorithms）sigmoid -> ReLU （激活函数使得刚开始的斜率直接为0，”梯度下降法“下降得更快，激活函数）=> 计算能力更强、提高迭代速度
   4. Idea => Experiment=> Code => Idea 不断循环用来训练神经网络模型

## B. Basics of Neural Network programming

1. 常用到的表示:
   1. (x,y)：表示一个单独的样本，其中x是nx维的特征向量，标签y值0或1
   2. 训练集是由m个训练样本构成。m=m_train
   3. (x<sup>(1)</sup>,y<sup>(1)</sup>)表示样本1
   4. X<sup>nxm</sup>由x1、x2.....列组成，有m列(python函数X.shape可输出矩阵维度)
   5. Y<sup>1xm</sup>=[y<sup>(1)</sup> y<sup>(2)</sup> y<sup>(3)</sup>......]
2. 二分分类
   1. 定义特征向量表示一张图片

## C. One hidden layer Neural Networks

## d. Deep Neural Networks

