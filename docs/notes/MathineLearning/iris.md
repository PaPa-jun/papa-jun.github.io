## 实验目的
本次实验旨在深入理解神经网络分类算法的基本原理，掌握Sigmoid激活函数的计算方法及前向传播和反向传播的过程，同时学会利用Python编程语言调用现有库实现神经网络分类算法，并能够手动构建简单的神经网络模型。

## 实验内容与步骤
1. **数据准备**
   - 使用Iris数据集作为实验数据源，该数据集包含了150个样本，每个样本具有4个特征值和1个类别标签。
   - 将数据集按80%:20%的比例划分为训练集和测试集。

2. **使用sklearn库实现**
   - 调用`sklearn.neural_network.MLPClassifier`构建多层感知机模型。
   - 对模型进行训练，并在测试集上评估其性能。

3. **手动实现神经网络**
   - 定义了一个名为`NeuralNetwork`的类，实现了Sigmoid激活函数及其导数、前向传播、反向传播等核心功能。
   - 通过调整学习率和迭代次数等超参数，训练模型并在测试集上进行预测，最终评估模型的准确性。

## 实验代码
- **使用sklearn库实现的代码片段**
  ```python
  from sklearn.neural_network import MLPClassifier
  from sklearn.preprocessing import StandardScaler
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
  from sklearn import datasets
  from pandas import DataFrame
  import numpy as np

  iris = datasets.load_iris()
  df = DataFrame(iris.data, columns=iris.feature_names)
  df["target"] = list(iris.target)
  X = df.iloc[:, 0:4]
  Y = df.iloc[:, 4]
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

  sc = StandardScaler()
  sc.fit(X)
  standard_train = sc.transform(X_train)
  standard_test = sc.transform(X_test)

  mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=123)
  mlp.fit(standard_train, Y_train)
  result = mlp.predict(standard_test)
  ```

  这一部分我首先利用 `train_test_split` 方法将数据集划分为 8:2，然后参考 sklearn 官方文档实现了模型的构建。
  
- **手动实现神经网络的代码片段**
  ```python
  class NeuralNetwork:
      def __init__(self, input_size, hidden_size, output_size):
          # 初始化权重和偏置
          ...
  
      def sigmoid(self, x):  # sigmoid 计算方式
          return 1 / (1 + np.exp(-x))
  
      def sigmoid_derivative(self, x):  # sigmoid 导数计算方式，输入为 x = sigmoid(input)
          return x * (1 - x)
  
      def forward(self, X):
          self.hidden_raw = np.dot(X, self.weights_input_hidden) + self.bias_hidden
          self.hidden = self.sigmoid(self.hidden_raw)
  
          # 前向传播计算输出层输出
          self.output_raw = np.dot(self.hidden, self.weights_hidden_output) + self.bias_output
          self.output = self.sigmoid(self.output_raw)
          return self.output
  
      def backward(self, X, y, output, learning_rate):
          # 计算输出层误差
          output_error = y - output
          output_delta = output_error * self.sigmoid_derivative(self.output)
  
          # 计算隐藏层误差
          hidden_error = output_delta.dot(self.weights_hidden_output.T)
          hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)
  
          # 更新权重和偏置
          self.weights_hidden_output += self.hidden.T.dot(output_delta) * learning_rate
          self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
          self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
          self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
  
      def train(self, X, y, epochs, learning_rate):
          # 模型训练
          ...
  
      def predict(self, X):
          # 模型预测
          ...
  ```
  
  这一部分要实现的主要是 Sigmoid 函数的导数计算，遵循公式：

$$
\sigma(x) = \frac{1}{1 + \exp(-x)} \implies \frac{\partial \sigma(x)}{\partial x} = \frac{\exp{(-x)}}{(1 + \exp{(-x)})^2} = \sigma(x)(1 - \sigma(x))
$$
  
  为了少计算两次 sigmoid，我在代码中用的逻辑是 `x * (1 - x)`，其中 `x = sigmoid(intput)`。
  
  另外就是根据反向传播算法更新参数：
  
  - 输出层的误差计算：$\delta_{\text{output}} = (\mathbf{a}_{\text{output}} - \mathbf{y}) \odot \sigma'(\mathbf{z}_{\text{output}})$ 
    - 其中 $\mathbf{y}$ 是真实标签，$\sigma'$ 是激活函数的导数隐藏层的误差计算
  
  - 隐藏层的误差：$\delta_{\text{hidden}} = (\delta_{\text{output}} \cdot \mathbf{W}_{\text{hidden-output}}^T) \odot \sigma'(\mathbf{z}_{\text{hidden}})$
  - 权重和偏置的更新
    - 隐藏层到输出层的权重更新：$\mathbf{W}_{\text{hidden-output}} \leftarrow \mathbf{W}_{\text{hidden-output}} - \eta \cdot \mathbf{a}_{\text{hidden}}^T \cdot \delta_{\text{output}}$
    - 输出层的偏置更新：$\mathbf{b}_{\text{output}} \leftarrow \mathbf{b}_{\text{output}} - \eta \cdot \delta_{\text{output}}$
    - 输入层到隐藏层的权重更新：$\mathbf{W}_{\text{input-hidden}} \leftarrow \mathbf{W}_{\text{input-hidden}} - \eta \cdot \mathbf{x}^T \cdot \delta_{\text{hidden}}$
    - 隐藏层的偏置更新：$\mathbf{b}_{\text{hidden}} \leftarrow \mathbf{b}_{\text{hidden}} - \eta \cdot \delta_{\text{hidden}}$


## 实验结果：调参与分析
- **使用sklearn库实现的结果**
  
  - 测试集上的预测准确率为1.0，表明模型在测试集上表现良好。
  - 模型结构为3层（输入层、隐藏层、输出层），迭代次数为982次，损失值为0.08276，输出层采用的是softmax激活函数。
  
- **手动实现神经网络的结果**
  
  ```yaml
  param1: 
  	hidden_size: 1
  	lr: 0.01
  	epoch: 500
  	accuracy: 0.4
  	loss: 0.057
  param2:
  	hidden_size: 10
  	lr: 0.01
  	epoch: 1000
  	accuracy: 1.0
  	loss: 0.007
  ```
  
  通过调整隐藏层大小，成功让模型准确率上升，Loss 下降。

## 结论
通过本次实验，我不仅掌握了如何使用sklearn库快速构建和评估神经网络模型，还深入了解了神经网络的工作机制，包括前向传播、反向传播等关键概念。此外，通过手动实现神经网络，进一步加深了对神经网络内部运作的理解。

## 反馈

- 建议：实验感觉没什么，但是手写作业貌似有很多地方表述不太清晰，希望可以明确一些；
- 用时：实验 2 h，纸质作业 8 h
