## 实验流程

本实验包括两个任务——回归和分类。对于每个任务，需要分别执行数据预处理、数据集划分、建立模型、设置优化器、训练模型以及评估模型这几个步骤。

### Regression Task

- 数据预处理：首先观察数据集的分布

  <img src="./oigin_data.png" width="80%">
  
  显然不符合线性回归模型的假设，即原始数据服从正态分布，因此对数据做对数变换，变换后分布
  
  <img src="./log_data.png" width="80%">
  
  可见训练数据分布接近正太分布；
  
- 数据集划分：利用 `train_test_split(test_size=0.1)` 将数据集划分为：训练集 + 验证集/测试集=9/1，返回 `Dataloader`；


- 建立模型：注册模型参数 `weights` 和 `bias`，前者可设置为全 $0$ 的 $(14\times 1)$ 的矩阵，后者根据处理后的训练数据分布，设置为均值为 $5.4$ 方差为 $1$ 的 $(1, 1)$ 的矩阵，完成向前传播算法即权重和输入以及偏置项的线性组合；

- 设置优化器：计算均方差损失 `mse_loss = np.mean((y_true.reshape(y_pred.shape) - y_pred)**2, axis=(0, 1))`，这里需要将输入的 `y_true` 的形状调整为列向量，否则会触发 `numpy` 的广播机制，使得求得损失为真实值的 $m$ 倍，计算梯度：

  ```py
  mse_grad = (-2 / batch_size) * np.dot(x.T, (y_true - y_pred))
  grads = {
      "weights": mse_grad,
  	"bias": np.mean(-2 * (y_true - y_pred), axis=0)
  }
  ```

- 训练模型：根据要求计算损失并更新参数；
- 模型评估：计算模型预测的相对误差 `relative_error = np.abs(pred_mean - true_mean) / true_mean`。

### Classification Task

- 数据预处理：出了对 `Run_time` 列做对数操作，还需要根据 `Run_time` 列的值和回归模型预测的均值创建 `label` 列并删除 `Run_time` 列；
- 数据集划分：利用 `train_test_split(test_size=0.1)` 将数据集划分为：训练集/测试集=9/1；
- 建立模型：注册模型参数 `beta`，融合了回归模型中的权重项和偏置项，为一个 $15 \times 1$ 的矩阵，完成向前传播，将 $Sigmoid$ 函数作用于输入向量与参数 `beta` 的线性组合，这里还要给输入向量增加一个维度并将其值设置为 $1$；
- 设置优化器：计算交叉熵损失 `-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))`，计算参数整体梯度 `np.dot(x.T, (y_pred - y_true)) / y_true.size`，这里也要对输入向量做升维处理；
- 训练模型：根据要求计算损失并更新参数；
- 模型评估：计算模型预测的正确率 $\frac{\lvert \mu - \mu_{true} \rvert}{\mu_{true}}$。

## 调参过程

### Regression Task

下面是最终训练的训练曲线：

<img src="./lossR.png" width="60%">

只要 `lr <= 4e-5`，基本都有这样的曲线，区别在于较大的 `every_decay` 会让曲线在前300个 epochs 内下降的速度稍微慢一些，较大的学习率，模型的 loss 降的更低一些，但是在图像中体现不明显。

我发现的本次实验的一个关键步骤是设置模型的初始参数（非超参数）。`weights` 如果设置的与 $0$ 有一定距离，会导致模型一开始的 Loss 非常大，大概在 $7000$ 左右，即使调整学习率到一个比较大的水平，最终 loss 也只能降低到两位数水平，设置为 $0$ 或者 $0$ 附近，则可以让 loss 降低到个位数。

`bias` 的初值设置则会较大地影响模型的性能，设置为 $0$ 时，最终 loss 最低有 $2.4$ 左右，$R^2$ 为 $-0.4$ 左右。我根据线性回归的一个默认假设：偏置服从正态分布，且模型预测结果也服从正态分布，且方差与偏置项方差一致。因此我绘制了原始数据与预测值的频率分布直方图对比：

<img src="./before.png" width="80%">

可见模型预测值中有很大一部分与原始数据不重合，接着我根据原始数据分布图将其设置成均值 $6.5$，方差接近 $1$ 的量，则模型 loss 降低至 $1.0$ 左右，且 $R^2$ 达到了 $0.36$ 左右。下面是预测和原始数据的频率分布直方图：

<img src="./after.png" width="80%">

可见调整之后预测结果与真实值重合度明显变高。

### Classification Task

下面是最终训练的训练曲线：

<img src="./lossC.png" width="60%">

该模型的损失下降较快，我只调整了 `lr` 和 `steps`，用一个接近阈值的学习率和较大的训练轮数完成了训练，保证模型损失能降低到一个比较好的水平。

## 最好结果

下面记录了我训练过程的最好结果以及对应参数。

### Regression Task

#### 参数

```yaml
batch_size: 4096
data_dir: SGEMM_GPU_performance
decay_every: 50
epochs: 100
in_features: 14
lr: 4.0e-05
lr_decay: 0.99
out_features: 1
results_path: Train
seed: 123
shuffle: true
task: Regression
```

#### 评测结果

```yaml
Mean Squared Error: 0.96512843440756
Average prediction: 6.277523272282774
Relative error: 0.0013091226991826669
R-squared: 0.36217277254261426
```

### Classification Task

#### 参数

```yaml
data_dir: SGEMM_GPU_performance
decay_every: 10
in_features: 14
lr: 0.002
lr_decay: 0.99
mean: 6.272240766668169
results_path: Train
seed: 123
steps: 10000
task: Classification
```

#### 评测结果

```yaml
Accuracy: 0.8543573829842255
```

### 与 SK-Learn 模型对比

SK-Learn 的回归模型 Loss 只有 $0.88$，相对误差 $0.1\%$，略微优于我的模型；分类模型准确率为 $83\%$，低于我的模型。

## 回答问题

### 额外操作

本次实验中我除了对数据 `Run_time` 列做了对数处理外没有做其他的额外操作。中间试过将前几列数据做归一化，但是效果并不好。此外我觉得并没有影响大的额外操作。

### 交叉熵损失的优势

- 凸函数：交叉熵在logistic 回归中是一个凸函数。 这使得我们只要找到局部最小值，则一定找到了全局最小值；
- 学习速度快：交叉熵损失函数不会受到 sigmoid 函数饱和区（接近 $0$ 和 $1$ 时）的影响，从而确保了更快的学习速度；
- 对数概率：交叉熵损失函数使用对数概率来衡量错误，这意味着当模型预测正确时，损失值较低；而当预测错误时，损失值较高。这种设计有助于放大那些预测错误的样本的影响，从而促使模型在训练过程中优先纠正这些错误；
- 增强模型稳定性：由于交叉熵损失函数在处理极端预测值时的表现更好，它有助于增强模型的稳定性，尤其是在处理不平衡数据集时。交叉熵损失函数能够更有效地应对极端情况，避免了模型在某些情况下过度自信的问题；
- 接近真实分布：在优化过程中，交叉熵损失函数实际上是在最小化 KL 散度，这表示模型试图使预测概率分布尽可能接近真实分布。

### 调参经验

- 学习率：从一个较小的学习率开始逐步增大，超过到某个阈值后，模型训练的损失会急剧上升，这表明到达了极限。此时可以选择保持哪个能够让损失稳定下降的最大的学习率。
- 训练轮数：根据训练损失曲线，选择一个能够让损失基本保持稳定的最大轮数即可。
- 学习率更新周期：较大的更新周期会使得模型的探索欲望较强，较小的更新周期使得模型较快趋于稳定，可以根据损失曲线的降低速度适当选择学习率更新周期；
- 学习率降低系数：该参数也决定了模型的探索欲望，较小的系数能让模型的损失下降得更慢，根据训练曲线选择；
- 批次大小：批次大小决定了更新模型的速率，越大模型收敛越快，但是太大容易造成震荡，也可以根据曲线选择。

### 正则化

本次实验我测试了正则化和不正则化的结果差别，发现变化不大，几乎没有影响，因此没有采用正则化。从原理的角度来看，本次实验也不需要正则化，原因如下：

- $\mathcal{L}_1$ 正则化的作用是稀疏矩阵，训练过程中，一些对预测目标影响不大的特征对应权重会被逐渐置零，因此采用 $\mathcal{L}_1$ 正则化能够起到选择特征的作用。在本次实验中，最后模型学习到的权重都在 $0$ 附近，且模型的特征并不多，因此该正则化的影响很小；
- $\mathcal{L}_2$ 正则化的作用是降低模型复杂度，使得模型权重不会太大。而本次实验模型学习到的权重本就不大，模型复杂性比较低，因此不需要考虑该正则化。

## 反馈

- 花费时间：大概 $5$ 天左右，第一天完成回归模型的代码部分，第二天对回归任务参数进行初步调整，第三天完成分类模型的代码任务，第四第五天联合调整回归任务和分类任务的参数；
- 建议：
  - 或许实验内容可以和上课内容结合地更紧密一些；
  - 希望可以再多一些理论讲解。