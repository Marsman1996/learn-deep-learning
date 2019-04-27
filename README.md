# MNIST <!-- omit in toc -->
[MINIST](http://deeplearning.net/tutorial/gettingstarted.html)是一个手写数字图片数据集, 本次实验使用MINIST作为数据集进行手写数字识别训练.

## Index <!-- omit in toc -->
- [导入数据](#导入数据)
- [建立模型](#建立模型)
  - [卷积层(C1)](#卷积层c1)
  - [池化层(S2)](#池化层s2)
  - [输入层(C3)](#输入层c3)
  - [池化层(S4)](#池化层s4)
  - [全连接层(F5)](#全连接层f5)
  - [Dropout层(D6)](#dropout层d6)
  - [全连接层(F7)](#全连接层f7)
- [训练模型](#训练模型)
- [评估模型](#评估模型)
- [附录: Tensorflow安装](#附录-tensorflow安装)

## 导入数据
训练集(train-images-idx3-ubyte)数据头部如表所示:  

|  offset  |      type      |      value       |    description    |
| :------: | :------------: | :--------------: | :---------------: |
|   0000   | 32 bit integer | 0x00000803(2051) |   magic number    |
|   0004   | 32 bit integer |      60000       | number of images  |
|   0008   | 32 bit integer |        28        |  number of rows   |
|   0012   | 32 bit integer |        28        | number of columns |
|   0016   | unsigned byte  |        ??        |       pixel       |
|   0017   | unsigned byte  |        ??        |       pixel       |
| ........ |
|   xxxx   | unsigned byte  |        ??        |       pixel       |

测试集(t10k-labels-idx1-ubyte)标签头部如图所示:  

| [offset] |     [type]     |     [value]      |      [description]       |
| :------: | :------------: | :--------------: | :----------------------: |
|   0000   | 32 bit integer | 0x00000801(2049) | magic number (MSB first) |
|   0004   | 32 bit integer |      10000       |     number of items      |
|   0008   | unsigned byte  |        ??        |          label           |
|   0009   | unsigned byte  |        ??        |          label           |
| ........ |
|   xxxx   | unsigned byte  |        ??        |          label           |

> The labels values are 0 to 9.

因此图像部分头部占16字节, 分别为魔数, 图片数量, 图片像素行, 图片像素列. 标签部分头部占8字节, 分别为魔数, 标签数量. 因此将提供的`.gz`压缩文件使用`gzip.open()`函数按字节流形式打开后按照上表将对应参数读取出来. 同时将读取出来的图片数据划分为验证集和数据集, 在这里设置验证集为5000张图片.

## 建立模型
此处使用的是一个简单的, 端到端的类[LeNet-5](http://yann.lecun.com/exdb/lenet/)卷积模型. 该模型如图所示  
![model](img/model.png)  
<!-- ### 输入层(INPUT) -->
### 卷积层(C1)

```python
self.conv1_weights = tf.Variable(
    tf.truncated_normal([5, 5, NUM_CHANNELS, 32],
    stddev=0.1,
    seed=SEED))
self.conv1_biases = tf.Variable(tf.zeros([32]))

conv = tf.nn.conv2d(data,
    self.conv1_weights,
    strides=[1, 1, 1, 1],
    padding='SAME')

relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
```

训练时输入数据`data`是 64\*28\*28\*1 向量(64张28\*28像素单通道图片). C1层卷积核为 5\*5\*1\*32 向量(有32个大小为5\*5的卷积核), 设置各方向步长为1且填充方式为`SAME`, 即得到卷积结果为 64\*26\*26\*32 向量(对64张图片每张图片都进行32个卷积核的运算). 然后为该结果中加上长度为64的1维数组`conv1_biases`后执行ReLU非线性激活, ReLU过程如下算式所示, 它只保留正数, 其他置0.
$$ 
Relu(x) = \begin{cases}x, & x < 0 \cr 
            0, &otherwise\end{cases}
$$

### 池化层(S2)

```python
pool = tf.nn.max_pool(relu,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
```

S2层池化窗口大小为2\*2, 窗口滑动步长为 2\*2, 则输出结果为 64\*14\*14\*32 的向量.

### 输入层(C3)

```python
self.conv2_weights = tf.Variable(
    tf.truncated_normal([5, 5, 32, 64], 
    stddev=0.1,
    seed=SEED))
self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

conv = tf.nn.conv2d(pool,
    self.conv2_weights,
    strides=[1, 1, 1, 1],
    padding='SAME')
relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))
```

类似C1层, C3层卷积核为 5\*5\*32\*64 向量, 得到卷积结果为 64\*12\*12\*64 的向量. 

### 池化层(S4)

```python
relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))
pool = tf.nn.max_pool(relu,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
```

类似S2层, S4层输出结果为 64\*7\*7\*64 向量.

### 全连接层(F5)

```python
self.fc1_weights = tf.Variable(
    tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
    stddev=0.1,
    seed=SEED))
self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

pool_shape = pool.get_shape().as_list()
reshape = tf.reshape(
    pool,
    [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
```

`get_shape()`函数获取维数后使用`tf.reshape()`函数将S4输出的结果转化为 64\*3136 的向量, 然后与`fc1_weights`矩阵相乘得到 64\*512 向量, 再偏置和ReLU非线性激活.

### Dropout层(D6)

```python
hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
```

以50%的概率丢弃神经元.

### 全连接层(F7)

```python
self.fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
    stddev=0.1,
    seed=SEED))
self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

tf.matmul(hidden, self.fc2_weights) + self.fc2_biases
```

类似F6层, 与512\*10的向量相乘, 输出 64\*10 向量.

## 训练模型

```python
# 损失计算（+L2正则化）
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=train_labels_node, logits=logits))
## L2正则化损失
regularizers = (tf.nn.l2_loss(model.fc1_weights) + tf.nn.l2_loss(model.fc1_biases) +
                tf.nn.l2_loss(model.fc2_weights) + tf.nn.l2_loss(model.fc2_biases))
## 总损失=样本损失+L2正则化损失
loss += 5e-4 * regularizers

# 优化器（用于参数更新）
## 设置一个每批增加一次的变量，并控制学习率衰减。
batch = tf.Variable(0)
## 每个时期衰减一次，使用从0.01开始的指数衰减。
learning_rate = tf.train.exponential_decay(
    0.01,                # Base learning rate.
    batch * BATCH_SIZE,  # Current index into the dataset.
    train_size,          # Decay step.
    0.95,                # Decay rate.
    staircase=True)
## 用momentum优化器优化
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

# 将数据送入计算图
feed_dict = {train_data_node: batch_data,
            train_labels_node: batch_labels}
# 运行优化器进行参数更新
sess.run(optimizer, feed_dict=feed_dict)
```

## 评估模型

```python
# 运行计算图得到评估信息
if step % EVAL_FREQUENCY == 0:
    l, lr, predictions = sess.run([loss, learning_rate, train_prediction], feed_dict=feed_dict)

# 测试输出结果模型准确率
test_error = error_rate(eval_in_batches(test_data, sess, eval_prediction, eval_data), test_labels)
```

运行时输出见 *log/2019-4-2-1815.log* . 最终训练出模型准确率为99.3%.

## 附录: Tensorflow安装
我是在Anaconda下进行tensorflow的安装的, 它可以方便的管理各种环境.

0. 需要安装[Anaconda](https://www.anaconda.com/), [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)和[CuDNN](https://developer.nvidia.com/rdp/cudnn-download)

    > 注意CUDA和CuDNN版本需要与你要安装的tensorflow版本相对应, 这里选择的是tensorflow-1.12, CUDA-9.0, CuDNN-7.1. 其他版本对应看[这个网站](https://www.tensorflow.org/install/source#tested_build_configurations).

1. `conda create -n tf1.12 python=3.6` 创建一个python版本为3.6的虚拟环境, 这是为了方便日后安装多个版本的tensorflow或是其他软件而不冲突.
2. `pip install tensorflow-gpu==1.12.0` 安装tensorflow