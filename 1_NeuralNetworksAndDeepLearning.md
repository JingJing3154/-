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
   
9. Logistic回归的梯度下降法

   1. exp：![image-20230117093802133](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230117093802133.png)<br>![image-20230117093830198](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230117093830198.png)<br>为了使得逻辑回归中最小化代价函数，我们需要做的仅仅是修改参数w和b的值，现在进行的就是反向计算的过程![image-20230117094456351](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230117094456351.png)![image-20230117095005280](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230117095005280.png)注意：该仅仅是一个样本的梯度下降法进行调整

10. m个样本的梯度下降法

    1. 代价函数=>针对于多个样本进行的综合（算术平均值）
    2. 针对于多个样本要进行对参数w<sub>1</sub>和w<sub>2</sub>以及b的综合优化（算术平均）![image-20230117100350676](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230117100350676.png)同时也要做到对微分的平均
    3. 需要注意的是，这仅仅是对两个参数进行的优化，就用到了两层for循环，如果多个参数时，会有更多层的循环，在应用深度学习算法的时候，你会发现在代码显式使用for循环会使算法很低效，并且在深度学习领域会有越来越大的数据集，所以能够应用算法并且没有显式的for循环是非常重要的，并且会帮助适用于更大的数据集，所以出现了向量化技术便于解决该类问题。

11. 向量化

    1. 运用向量化的原因：运行速度会更快（cpu以及gpu的并行决定了这一点）
    2. 一般会使用python内置的numpty内置函数帮助计算，所以当你在写循环的时候可以检查numpty是否存在类似的内置函数从而避免使用loop。

12. 向量化Logistic回归

    1. 训练输入 X => (n<sub>x</sub>，m)的矩阵 R<sup>n<sub>x</sub>*m</sup>
    2. 计算公式: [z<sup>(1)</sup>z<sup>(2)</sup>.....z<sup>(m)</sup>] = w<sup>T</sup>X + [bbbbbb] = [w<sup>T</sup>x<sup>(1)</sup> + b,w<sup>T</sup>x<sup>(2)</sup> + b,w<sup>T</sup>x<sup>(m)</sup> + b,]
    3. 由上述z综合组成的1*m矩阵综合成为Z的变量
    4. 接下来的任务就是找到一个同时计算[a<sup>(1)</sup>,a<sup>(2)</sup>...a<sup>(m)</sup>]的方法，利用σ一次性计算所有的a
    
13. 向量化Logistic 回归的梯度输出（同时计算m个数据的梯度，并且实现一个非常高效的逻辑回归算法Logistic Regression）

    1. 之前梯度计算中 dz<sup>(1)</sup> = a<sup>(1)</sup> - y<sup>(1)</sup>,dz<sup>(2)</sup> = a<sup>(2)</sup> - y<sup>(2)</sup>.....现在对于m个训练数据做同样的运算，我们可以定义一个新的变量dZ 即所有的dz变量横向排列
    2. 我们已经知道所有的dz<sup>(i)</sup>已经组成了一个行向量dZ，在python中，我们很容易联想到db = np.sum(dZ)/m，dw  = X*dz<sup>T</sup>,这样的话能够避免在训练集上使用for循环
    3. ![image-20230202172850905](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230202172850905.png) ![image-20230202172755213](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230202172755213.png)

14. python中的广播（介绍函数使用）

    1. X.sum(axis) 其中sum的参数表示求和运算需要按列执行（0轴是垂直的，即列。而1轴是水平的，也就是行）
    2. X.reshape 调用了numpy中的广播机制，当我们写代码不确定矩阵维度的时候，通常会使用重塑来确保得到我们想要的列向量或者行向量。（注：reshape是一个常量时间的操作，时间复杂度是O（1））实际上就是对原来矩阵的一种扩充

15. python中的一些问题：

    1. shape 中 (5,) 是指一个一维数组而不是一个矩阵，其转置也是一个一维数组。同时输出a和a<sup>T</sup>的内积，只会得到一个数。
    2. 运行的命令是np.random.randn(5)会生成一个一维数组，一般会使用（5，1）column shape或者（1，5）row shape
    3. 不确定一个向量的维度时，经常会使用一个assert

16. 习题总结

    1. 神经元计算一个线性函数，然后接一个激活函数
    
    2. 编程练习
    
       

## C. One hidden layer Neural Networks

1. 神经网络概述
   1. 可以把许多sigmoid单元堆叠起来形成一个神经网络![image-20230206103937477](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230206103937477.png)s首先计算第一层网络中的各个节点的相关数z，接着计算a，在下一层网络同理，会使用符号<sup>[m]</sup>表示第m层网络中节点相关的数，节点的集合被称为第m层网络计算过程如下：![image-20230206104721129](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230206104721129.png)![image-20230206104734297](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230206104734297.png)![image-20230206104739529](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230206104739529.png)![image-20230206104948299](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230206104948299.png)
   2. 神经网络表示：![image-20230206105236744](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230206105236744.png)
      1. 我们有输入特征x1，x2，x3，它们被竖直的堆叠，这叫做神经网络的输入层，它包含了神经网络的输入，中间这一层我们称之为隐藏层。本例中最后一层只由一个结点构成，只有一个结点的层被称为输出层，它负责产生预测值。**隐藏层解释：在一个神经网络中，监督学习训练的时候，训练集包括了输入x和输出y，但中间结点的准确值我们是不知道的，我们看不见它们在训练集中应有的值。
      2. 引入符号：a<sup>[0]</sup>可用来表示输入特征，a表示激活的意思，它意味着网络中不同层的值会传递到它们后面的层中，输入层将x传递给隐藏层，所以我们将输入层的激活值称为a<sup>[0]</sup>，下一层隐藏层同样会产生一些激活值，我们将其记作a<sub>1</sub><sup>[1]</sup>，这样的解释就是第一层的第一个结点或是第一个单元。
      3. 就该例而言，最后产生的数值a，它只是一个单独的实数，所以将最后的输出值也可以取为a<sup>[2]</sup>。这与逻辑回归相似。
      4. 我们在计算网络的层数时，输入层是不算入总层数之中的，所以隐藏层是第一层，习惯上将输入层称为第0层
      5. 我们看到的隐藏层以及最后的输出层是带有参数的。这里的隐藏层将会拥有两个参数W和b，注意：参数也会有上标的，即用来表示某层的参数，W矩阵的行列数与输入和单元有关![image-20230206110742307](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230206110742307.png)
   3. 神经网络的输出
      1. 逻辑回归的计算有两个步骤，首先按照步骤计算出z，以sigmod函数为激活函数计算z（得出a），一个神经网络只是这样子做了好几次重复计算。![image-20230206113349556](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230206113349556.png)
      2. 向量化计算：将四个等式向量化之后，向量化的过程是将网络中的一层神经元参数纵向堆积起来，例如隐藏层中的w纵向堆积起来变成一个（4，3）的矩阵，用符号W<sup>[1]</sup>表示。另一个视角是我们由四个逻辑回归单元，且每一个逻辑回归单元都有相应的参数-向量W，将四个向量堆积在一起，会得出4x3的矩阵，由公式：![image-20230206165348002](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230206165348002.png)![image-20230206165353191](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230206165353191.png)![image-20230206165402729](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230206165402729.png)针对于每一层的输入以及输出，后一层的表示同样可以写成类似的形式。
      3. 该层W的行列由前一层和后一层共同决定，其中行数是这一层的单元个数，列数是上一层的单元个数
      4. ![image-20230206170151379](D:\GitHubResourse\复试内容准备\机器学习\1_NeuralNetworksAndDeepLearning.assets\image-20230206170151379.png)
   4. 

## d. Deep Neural Networks

