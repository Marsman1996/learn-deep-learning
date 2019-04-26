import tensorflow as tf
import gzip
import sys
import struct
import numpy
import time

train_images_filename = "data/train-images-idx3-ubyte.gz"
train_labels_filename = "data/train-labels-idx1-ubyte.gz"
t10k_images_filename  = "data/t10k-images-idx3-ubyte.gz"
t10k_labels_filename  = "data/t10k-labels-idx1-ubyte.gz"

NUM_CHANNELS = 1
PIXEL_DEPTH = 255
BATCH_SIZE = 64
NUM_LABELS = 10
EVAL_BATCH_SIZE = 64
SEED = 66478  # Set to None for random seed.
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
VALIDATION_SIZE = 5000  # 验证集大小
NUM_EPOCHS = 10

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS

IMAGE_SIZE = 0

def read_images(filename):
    with gzip.open(filename) as bytestream:
        # bytestream.read(16) #每个像素存储在文件中的大小为16bits
        # magic = int.from_bytes(bytestream.read(4), byteorder='big')
        bytestream.read(4)
        num_images = int.from_bytes(bytestream.read(4), byteorder='big')
        num_rows = int.from_bytes(bytestream.read(4), byteorder='big')
        num_columns = int.from_bytes(bytestream.read(4), byteorder='big')
        global IMAGE_SIZE
        IMAGE_SIZE = num_rows

        buf = bytestream.read(num_columns * num_rows * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32) 
        #像素值[0, 255]被调整到[-0.5, 0.5]
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        #调整为4维张量[image index, y, x, channels]
        data = data.reshape(num_images, num_columns, num_rows, NUM_CHANNELS)
        return data

def read_labels(filename):
    with gzip.open(filename) as bytestream:
        # bytestream.read(8) #每个标签存储在文件中的大小为8bits
        # magic = int.from_bytes(bytestream.read(4), byteorder='big')
        bytestream.read(4)
        num_images = int.from_bytes(bytestream.read(4), byteorder='big')

        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        return labels

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == labels) /
        predictions.shape[0])

class Model:
    def __init__(self):
        self.conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                                stddev=0.1,
                                seed=SEED))
        self.conv1_biases = tf.Variable(tf.zeros([32]))
        self.conv2_weights = tf.Variable(tf.truncated_normal(
            [5, 5, 32, 64], stddev=0.1,
            seed=SEED))
        self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
        self.fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                                stddev=0.1,
                                seed=SEED))
        self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
        self.fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                        stddev=0.1,
                                                        seed=SEED))
        self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    def build_model(self, data, train=False):
        conv = tf.nn.conv2d(data,
                            self.conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # 偏置和ReLU非线性激活。
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
        # 最大池化。
        # 内核大小规范{ksize}也遵循数据布局。 这里我们有一个2的池化窗口和2的步幅。
        pool = tf.nn.max_pool(relu,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
        conv = tf.nn.conv2d(pool,
                            self.conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))
        pool = tf.nn.max_pool(relu,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
        # 将特征图变换为2D矩阵，以将其提供给完全连接的图层。
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # 全连接层。
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
        # 在训练时，添加dropout层。
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, self.fc2_weights) + self.fc2_biases

def eval_in_batches(data, sess, eval_prediction, eval_data):
    """Get all predictions for a dataset by running it in small batches."""
    # global eval_prediction, eval_data
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in range(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={eval_data: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

def main(argv=None):
    if(FLAGS.self_test == True):
        pass
    else:
        # 加载图像
        train_data = read_images(train_images_filename)
        test_data  = read_images(t10k_images_filename)
        # 加载标签
        train_labels = read_labels(train_labels_filename)
        test_labels  = read_labels(t10k_labels_filename)
        # 
        validation_data = train_data[:VALIDATION_SIZE, ...]
        validation_labels = train_labels[:VALIDATION_SIZE]
        train_data = train_data[VALIDATION_SIZE:, ...]
        train_labels = train_labels[VALIDATION_SIZE:]
        num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]
    
    # 创建输入占位符
    train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    eval_data = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    # 初始化变量
    model = Model()

    # CNN模型构建
    logits = model.build_model(train_data_node, True)

    # 训练与评估
    # 损失计算（+L2正则化）
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=train_labels_node, logits=logits))
    # L2正则化损失
    regularizers = (tf.nn.l2_loss(model.fc1_weights) + tf.nn.l2_loss(model.fc1_biases) +
                    tf.nn.l2_loss(model.fc2_weights) + tf.nn.l2_loss(model.fc2_biases))
    # 总损失=样本损失+L2正则化损失
    loss += 5e-4 * regularizers

    # 优化器（用于参数更新）
    # 设置一个每批增加一次的变量，并控制学习率衰减。
    batch = tf.Variable(0)
    # 每个时期衰减一次，使用从0.01开始的指数衰减。
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    # 用momentum优化器优化
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

    train_prediction = tf.nn.softmax(logits)
    eval_prediction = tf.nn.softmax(model.build_model(eval_data))

    # 创建会话训练、评估模型
    # saver = tf.train.Saver(max_to_keep=1)
    start_time = time.time()
    with tf.Session() as sess:
        # 初始化可训练变量
        tf.global_variables_initializer().run()
        for step in range(int(num_epochs * train_size) // BATCH_SIZE):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            # 将数据送入计算图
            feed_dict = {train_data_node: batch_data,
                        train_labels_node: batch_labels}
            # 运行优化器进行参数更新
            sess.run(optimizer, feed_dict=feed_dict)
            # 打印信息
            if step % EVAL_FREQUENCY == 0:
                # 运行计算图得到评估信息
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                            feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                    (step, float(step) * BATCH_SIZE / train_size,
                    1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' % error_rate(
                    eval_in_batches(validation_data, sess, eval_prediction, eval_data), validation_labels))
                sys.stdout.flush()
        # 输出测试结果
        test_error = error_rate(eval_in_batches(test_data, sess, eval_prediction, eval_data), test_labels)
        print('Test error: %.1f%%' % test_error)
        print("Save model")
        # saver.save(sess, "model/mnist", global_step=step)
        builder = tf.saved_model.builder.SavedModelBuilder("model")
        builder.add_meta_graph_and_variables(sess, ['mnist'])
        builder.save()
        if FLAGS.self_test:
            print('test_error', test_error)
            assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
                test_error,)

if (__name__ == "__main__"):
    tf.app.run()