import tensorflow as tf

def loadmodel(argv=None):
    sess = tf.Session()
    print("load data")
    with tf.Session(graph=tf.Graph()) as sess:
        print("restore data")
        tf.saved_model.loader.load(sess, ['mnist'], "model")
    # print(conv1_weights)


if (__name__ == "__main__"):
    tf.app.run(loadmodel)