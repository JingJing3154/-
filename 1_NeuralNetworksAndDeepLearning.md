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
   5. Y<sup>1xm</sup>=[y<sup>(1)</sup> y<sup>(2)</sup> y<sup>(3)</sup>......] Y.shape = (1,m)
   6. ![image-20230116100801672](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230116100801672.png)

2. 二分分类(主要说结果 1|0)

   1. 定义特征向量表示一张图片

3. logistic回归

   1. 结果介于0-1之间，所以一般不能够使用线性函数进行预测，所以要使用sigmod()函数**0-1的光滑曲线**，y=sigmod(w<sup>T</sup>x + b) 参数里面还有b，这是一个实数（表示偏差）
   2. ![image-20230116100229113](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230116100229113.png)

4.  Logistic 回归损失函数 (Logistic Regression Cost Function)

   1. 成本函数：衡量算法的运行情况，来衡量预测输出值和实际值有多么接近。

   2. 在逻辑回归种常用到的损失函数是：该函数能够保证
      $$
      y \quad and\quad \overline{y} \quad 能够保持接近的状态
      $$
      ![image-20230116101015115](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230116101015115.png)<br>这种函数计算的是单个样本和预测值之间的接近程度

   3. 代价函数-> 所有训练集的训练样本的平均程度![image-20230116102625545](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230116102625545.png)

   4. 因此我们在训练逻辑回归模型时，需要找到合适的W和b，使得代价函数的J的总代价降到最低。、

5. 梯度下降法

   1. 形象化说明![image-20230116105325806](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230116105325806.png)
   2. 在实践中w可以是更高的维度，我们应用到函数中的任意w、b都是图中的某一点，在该点曲面的高度J是某一点的函数值，我们希望不断地调整w、b使得让该点的高度降低到最低点使得成本（代价）降到最低。
   3. 注意：代价函数\成本函数需要是一种凸函数
   4. 优化的过程就是**朝最陡的下坡方向走，不断的迭代。直到走到全局最优解或者接近全局最优解的地方**
   5. 单个参数的梯度下降法优化![image-20230116112702038](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230116112702038.png)<br>:=表示更新参数 a表示学习率（learning rate），用来控制步长（step），即向下走一步的长度，以及导数
   6. 双参数的梯度下降法<br>![image-20230116112907031](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230116112907031.png)

6. 导数 =>变化率

7. 计算图

   1. 神经网络的计算，都是按照前向或者反向传播过程组织的。首先我们计算出一个新的网络的输出（前向），紧接着进行一个反向传输操作。后者我们用来计算出相应的梯度和导数。
   2. 计算图：就是将计算的式子进行逐步翻译，中缀表达式，每一次运算结果用另一参数表示，便于反向计算时对参数的调整。
   3. ![image-20230116155131310](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230116155131310.png)
      1. 从左到右，正常的计算步骤，计算出J
      2. 从右到左，用于计算导数最自然的方式

8. 计算图的导数计算（链式法则）（反向传播计算）

   1. ![image-20230116155508686](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230116155508686.png)</br><center>这是一个流程图</center>
   2. 链式计算公式：![image-20230116155815246](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230116155815246.png)
   3. 输出变量对于某个变量的导数，我们用dvar命名

## C. One hidden layer Neural Networks

## d. Deep Neural Networks

